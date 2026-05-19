import triton
import torch

# Triton kernel for RMS normalization
@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, x_batch_stride, x_m_stride, x_n_stride,
                   rms_w_n_stride, output_batch_stride, output_m_stride, output_n_stride,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    # to get the right pointer
    offset_batch = pid_batch * x_batch_stride
    offset_m = pid_m * x_m_stride
    rms_weight_n_offset = pid_m * rms_w_n_stride
    
    # Initialise the sum of squares to zero
    _sum_sq = tl.zeros([BLOCK_N_SIZE], dtype=tl.float32)
    
    # Iterate over chunks of N_SIZE elements
    for i in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = i + tl.arange(0, BLOCK_N_SIZE)
        x_ptrs = x_ptr + offset_batch + offset_m + offset_n * x_n_stride

        x = tl.load(x_ptrs)
        _sum_sq += x.to(tl.float32) * x.to(tl.float32)
    
    # Calculate variance and standard deviation
    var = _sum_sq / N_SIZE
    rstd = tl.math.rsqrt(var + eps)
    
    # Normalize and scale input
    for i in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = i + tl.arange(0, BLOCK_N_SIZE)
        x_ptrs = x_ptr + offset_batch + offset_m + offset_n * x_n_stride
        rms_weight_ptrs = rms_w_ptr + rms_weight_n_offset + offset_n
        
        x = tl.load(x_ptrs)
        rms_weight = tl.load(rms_weight_ptrs)
        
        normalized_x = x * rstd
        output = normalized_x * rms_weight
        
        # write output back to DRAM
        output_ptrs = output_ptr + offset_batch + offset_m + offset_n * output_n_stride
        tl.store(output_ptrs, output)

# Wrapper function to call Triton kernel
@torch.inference_mode()
def rmsnorm_triton_wrapper(x, rms_w):
    BATCH_SIZE, M, N = x.shape
    output = torch.empty_like(x)
    rmsnorm_triton[(BATCH_SIZE, M)](x, rms_w, output,
                                      x.stride(0), x.stride(1), x.stride(2),
                                      rms_w.stride(0),
                                      output.stride(0), output.stride(1), output.stride(2),
                                      N_SIZE=N, eps=1e-6)
    return output
