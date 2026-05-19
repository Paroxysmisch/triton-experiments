import triton
import triton.language as tl

@triton.jit
def _logsumexp_kernel(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(N, BLOCK_SIZE)
    
    # Initialize variables
    alpha = -float('inf')
    res = 0.0
    
    # Iterate over blocks
    for i in range(grid_size):
        # Load data into shared memory
        x_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        x_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_mask = x_idx < N
        x_shared = tl.load(X + x_idx * xn_stride, mask=x_mask, other=-float('inf'))
        
        # Compute local maximum
        c = tl.max(x_shared, axis=0)
        
        # Update global maximum and adjust res
        alpha_new = tl.maximum(c, alpha)
        exp_diff = tl.exp(c - alpha_new)
        res += exp_diff
        
        # Synchronize across threads
        tl.barrier()
        
        # Update alpha
        alpha = alpha_new
    
    # Store the result
    res = tl.log(res) + alpha
    tl.store(OUT + pid * out_stride, res)
