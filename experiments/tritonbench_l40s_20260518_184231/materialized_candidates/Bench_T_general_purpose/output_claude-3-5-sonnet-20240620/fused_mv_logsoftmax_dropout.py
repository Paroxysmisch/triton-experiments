import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(A, v, out, p, n, m, dim, training):
    # Compute matrix-vector multiplication
    row = tl.program_id(0)
    if row < n:
        z = 0.0
        for j in range(m):
            z += A[row, j] * v[j]
        
        # Log-softmax computation
        max_z = tl.max(z)  # For numerical stability
        exp_z = tl.exp(z - max_z)
        log_sum_exp = tl.log(tl.sum(exp_z))
        s = z - max_z - log_sum_exp
        
        # Apply dropout
        if training:
            mask = tl.rand(tl.shape(s)) > p
            out[row] = s * mask
        else:
            out[row] = s

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None) -> Tensor:
    n, m = input.shape
    assert input.shape[1] == vec.shape[0], "Input and vector shapes are incompatible for matrix-vector multiplication."
    
    if out is None:
        out = torch.empty(n, dtype=input.dtype, device=input.device)
    
    # Launch the Triton kernel
    grid = (n,)
    fused_mv_logsoftmax_dropout_kernel[grid](input, vec, out, p, n, m, dim, training)
    
    return out
