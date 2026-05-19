import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    out_ptr,
    x_batch_stride,
    x_m_stride,
    x_k_stride,
    out_batch_stride,
    out_m_stride,
    out_k_stride,
    M: tl.constexpr,
    K: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    # Parallelize over batch and M dimensions
    batch_idx = tl.program_id(0)
    m_idx = tl.program_id(1)

    # Compute the base pointer for the current batch and M index in x
    x_batch_offset = batch_idx * x_batch_stride
    x_m_offset = m_idx * x_m_stride
    x_row_ptr = x_ptr + x_batch_offset + x_m_offset

    # Initialize sum of squares
    sum_sq = 0.0
    # Loop over K in blocks to compute sum of squares
    for k_offset_start in range(0, K, BLOCK_N_SIZE):
        k_offsets = k_offset_start + tl.arange(0, BLOCK_N_SIZE)
        mask = k_offsets < K
        x_ptrs = x_row_ptr + k_offsets * x_k_stride
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        x_sq = x_vals * x_vals
        sum_sq += tl.sum(x_sq, axis=0)

    # Compute RMS with epsilon for numerical stability
    rms = tl.sqrt(sum_sq / K + eps)
    inv_rms = 1.0 / rms  # Precompute inverse for efficiency

    # Loop over K in blocks to apply normalization and weights
    for k_offset_start in range(0, K, BLOCK_N_SIZE):
        k_offsets = k_offset_start + tl.arange(0, BLOCK_N_SIZE)
        mask = k_offsets < K
        # Load x and weights
        x_ptrs = x_row_ptr + k_offsets * x_k_stride
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        w_ptrs = rms_w_ptr + k_offsets
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0)
        # Normalize and scale
        normalized = x_vals * inv_rms
        scaled = normalized * w_vals
        # Compute output pointers and store
        out_batch_offset = batch_idx * out_batch_stride
        out_m_offset = m_idx * out_m_stride
        out_ptr_base = out_ptr + out_batch_offset + out_m_offset
        out_ptrs = out_ptr_base + k_offsets * out_k_stride
        tl.store(out_ptrs, scaled, mask=mask)

def rmsnorm_wrapper(x, rms_weight, eps=1e-6):
    # Ensure input is 3D: [batch, M, K]
    assert x.dim() == 3, "Input tensor must be 3-dimensional"
    batch, M, K = x.shape
    assert rms_weight.shape == (K,), "RMS weight must match last dimension of input"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Retrieve strides for input and output
    x_batch_stride = x.stride(0)
    x_m_stride = x.stride(1)
    x_k_stride = x.stride(2)
    
    out_batch_stride = output.stride(0)
    out_m_stride = output.stride(1)
    out_k_stride = output.stride(2)
    
    # Determine block size for K dimension
    BLOCK_N_SIZE = triton.next_power_of_2(K)
    if BLOCK_N_SIZE > 1024:
        BLOCK_N_SIZE = 1024  # Limit to maximum block size
    
    # Define grid dimensions (batch, M)
    grid = (batch, M)
    
    # Launch the Triton kernel
    rmsnorm_triton[grid](
        x, rms_weight, output,
        x_batch_stride, x_m_stride, x_k_stride,
        out_batch_stride, out_m_stride, out_k_stride,
        M=M, K=K, eps=eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
        num_warps=8  # Adjust based on block size and hardware
    )
    
    return output
