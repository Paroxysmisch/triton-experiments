import triton
import triton.language as tl
import torch

# Triton kernel for fused repeat interleave and log-softmax
@triton.jit
def _fused_repeat_interleave_log_softmax(X, OUT, repeats, dim, stride_in, stride_out, N, BLOCK_N: tl.constexpr):
    # Calculate the block index
    rm = tl.program_id(0)
    
    # Initialize the output block
    out_block = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Iterate over the input elements with repeats
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        mask = rn < N
        
        # Load the input elements
        X_idx = rm * stride_in + rn
        x = tl.load(X + X_idx, mask=mask, other=-float('inf'))
        
        # Apply repeat interleave
        repeat_counts = tl.load(repeats + rn, mask=mask, other=0)
        repeated = tl.broadcast(x, repeat_counts)
        
        # Compute max for numerical stability
        max_val = tl.max(repeated, axis=0)
        
        # Compute exponentials and sum
        exp_values = tl.exp(repeated - max_val)
        sum_exp = tl.sum(exp_values, axis=0)
        
        # Compute log-softmax
        log_softmax = repeated - max_val - tl.log(sum_exp)
        
        # Store the result
        OUT_idx = rm * stride_out + rn
        tl.store(OUT + OUT_idx, log_softmax, mask=mask)

# Wrapper function
def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    assert input.is_cuda
    if dim is None:
        input = input.flatten()
        dim = 0
    
    # Calculate the output size if not provided
    if output_size is None:
        output_size = sum(repeats)
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty(output_size, dtype=dtype or input.dtype, device=input.device)
    
    # Determine the strides
    stride_in = input.stride(dim)
    stride_out = out.stride(dim)
    
    # Determine the number of elements along the dimension
    N = input.size(dim)
    
    # Launch the Triton kernel
    M = input.numel() // N
    _fused_repeat_interleave_log_softmax[(M,)](
        input, out, repeats, dim, stride_in, stride_out, N, BLOCK_N=1024, num_warps=4
    )
    
    return out
