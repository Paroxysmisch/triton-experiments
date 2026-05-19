import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3_ptr, primals_1_ptr, primals_2_ptr,
    buf0_ptr, buf3_ptr, buf4_ptr,
    S, D,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Compute offset for the current row
    offset = row_idx * D + col_idx
    
    # Load input data
    primals_3 = tl.load(primals_3_ptr + offset, mask=col_idx < D, other=0.0)
    
    # Compute mean
    mean = tl.sum(primals_3, axis=0) / D
    tl.store(buf0_ptr + row_idx, mean)
    
    # Compute variance
    var = tl.sum((primals_3 - mean) ** 2, axis=0) / D
    
    # Compute normalized output
    norm_output = (primals_3 - mean) / tl.sqrt(var + 1e-5)
    
    # Store normalized output
    tl.store(buf4_ptr + offset, norm_output, mask=col_idx < D)
    
    # Store auxiliary output (e.g., variance or any other necessary computation)
    tl.store(buf3_ptr + row_idx, var)


import torch

def fused_native_layer_norm(primals_1, primals_2, primals_3):
    S, D = primals_3.shape
    
    # Allocate buffers
    buf0 = torch.empty((S,), dtype=primals_3.dtype, device=primals_3.device)
    buf3 = torch.empty((S,), dtype=primals_3.dtype, device=primals_3.device)
    buf4 = torch.empty_like(primals_3)
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Example block size, tune this for your hardware
    grid = (S,)
    
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, primals_1, primals_2,
        buf0, buf3, buf4,
        S, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return buf4, primals_3, buf0, buf3
