import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    # Pointers to tensors
    primals_3_ptr, out_ptr1, primals_1_ptr, primals_2_ptr,
    out_ptr0_mean, out_ptr0_var,
    # Tensor dimensions
    S, D,
    # Strides
    stride_primals_3, stride_out1,
    # Parameters
    eps: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr, RBLOCK: tl.constexpr,
):
    # Each block processes a single row
    row_idx = tl.program_id(0)
    if row_idx >= S:
        return
    
    row_start = row_idx * stride_primals_3
    output_row_start = row_idx * stride_out1
    
    # Thread index within the block
    tid = tl.arange(0, BLOCK_SIZE)
    
    # Shared memory for Welford reduction
    sh_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sh_m2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sh_weight = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Initialize thread-local statistics
    thread_mean = 0.0
    thread_m2 = 0.0
    thread_weight = 0
    
    # Loop over elements in chunks of RBLOCK * BLOCK_SIZE
    for offset in range(0, D, RBLOCK * BLOCK_SIZE):
        cols = offset + tid * RBLOCK + tl.arange(0, RBLOCK)
        mask = cols < D
        x = tl.load(primals_3_ptr + row_start + cols, mask=mask, other=0.0)
        
        # Welford's algorithm for this chunk
        for element in tl.ravel(x):
            thread_weight += 1
            delta1 = element - thread_mean
            thread_mean += delta1 / thread_weight
            delta2 = element - thread_mean
            thread_m2 += delta1 * delta2
    
    # Store thread stats in shared memory
    sh_mean[tid] = thread_mean
    sh_m2[tid] = thread_m2
    sh_weight[tid] = thread_weight
    tl.barrier()
    
    # Tree reduction to combine stats
    stride = 1
    while stride < BLOCK_SIZE:
        if tid % (2 * stride) == 0:
            other = tid + stride
            if other < BLOCK_SIZE:
                a_mean = sh_mean[tid]
                a_m2 = sh_m2[tid]
                a_weight = sh_weight[tid]
                b_mean = sh_mean[other]
                b_m2 = sh_m2[other]
                b_weight = sh_weight[other]
                
                total_weight = a_weight + b_weight
                delta = b_mean - a_mean
                combined_mean = a_mean + delta * b_weight / total_weight
                combined_m2 = a_m2 + b_m2 + delta**2 * a_weight * b_weight / total_weight
                
                sh_mean[tid] = combined_mean
                sh_m2[tid] = combined_m2
                sh_weight[tid] = total_weight
        stride *= 2
        tl.barrier()
    
    # Final mean and variance
    mean = sh_mean[0]
    variance = sh_m2[0] / sh_weight[0]
    inv_std = tl.sqrt(1.0 / (variance + eps))
    
    # Normalize and apply affine transformation
    for offset in range(0, D, RBLOCK * BLOCK_SIZE):
        cols = offset + tid * RBLOCK + tl.arange(0, RBLOCK)
        mask = cols < D
        x = tl.load(primals_3_ptr + row_start + cols, mask=mask)
        normalized = (x - mean) * inv_std
        
        if primals_1_ptr is not None:
            scale = tl.load(primals_1_ptr + cols, mask=mask, other=1.0)
            normalized *= scale
        if primals_2_ptr is not None:
            bias = tl.load(primals_2_ptr + cols, mask=mask, other=0.0)
            normalized += bias
        
        tl.store(out_ptr1 + output_row_start + cols, normalized, mask=mask)
    
    # Store global stats
    if tid == 0:
        tl.store(out_ptr0_mean + row_idx, mean)
        tl.store(out_ptr0_var + row_idx, variance)

# Wrapper Function
def fused_native_layer_norm(primals_3, gamma=None, beta=None, eps=1e-5):
    S, D = primals_3.shape
    device = primals_3.device
    
    # Allocate output buffers
    buf4 = torch.empty_like(primals_3)
    buf0 = torch.empty(S, device=device, dtype=primals_3.dtype)
    buf3 = torch.empty(S, device=device, dtype=primals_3.dtype)
    
    # Kernel configuration
    BLOCK_SIZE = min(triton.next_power_of_2(D // 4), 1024)
    RBLOCK = 1  # Elements per thread iteration
    
    grid = (S,)
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, buf4, gamma, beta, buf0, buf3,
        S, D,
        primals_3.stride(0), buf4.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE, RBLOCK=RBLOCK,
        num_warps=4  # Adjust based on BLOCK_SIZE
    )
    
    return buf4, buf0, buf3
