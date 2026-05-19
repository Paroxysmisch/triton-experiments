import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer_type,
    rms_w_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    stride_m: tl.int32,
    n_cols: tl.int32,
    eps: tl.float32,
    BLOCK_N_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers for current row
    row_start_ptr = x_ptr + row_idx * stride_m
    out_start_ptr = output_ptr + row_idx * stride_m
    
    # Create offsets for the block
    offs = tl.arange(0, BLOCK_N_SIZE)
    mask = offs < n_cols
    
    # Load input row
    x = tl.load(row_start_ptr + offs, mask=mask, other=0.0)
    w = tl.load(rms_w_ptr + offs, mask=mask, other=1.0)
    
    # Compute RMS norm
    x_squared = x * x
    mean_squared = tl.sum(x_squared, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_squared + eps)
    
    # Normalize and scale
    out = x * inv_rms * w
    
    # Store result
    tl.store(out_start_ptr + offs, out, mask=mask)

def rmsnorm_triton_wrapper(x: torch.Tensor, rms_weight: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper function for RMS normalization using Triton kernel
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        rms_weight: Weight vector of shape (hidden_dim,)
        eps: Small constant for numerical stability
    """
    assert x.is_cuda and rms_weight.is_cuda, "Input tensors must be on GPU"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    batch_size, seq_len, hidden_dim = x.shape
    n_rows = batch_size * seq_len
    x_reshaped = x.view(n_rows, hidden_dim)
    
    # Output tensor
    output = torch.empty_like(x_reshaped)
    
    # Launch configs
    BLOCK_N = triton.next_power_of_2(hidden_dim)
    grid = (n_rows,)
    
    # Launch kernel
    rmsnorm_triton[(grid,)](
        x_reshaped,
        rms_weight,
        output,
        x_reshaped.stride(0),
        hidden_dim,
        eps,
        num_warps=4,
        BLOCK_N_SIZE=BLOCK_N,
    )
    
    return output.view_as(x)
