import triton
import triton.language as tl
import torch
import torch.nn.functional as F

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    X_ptr, Y_ptr, Z_ptr, gamma_ptr,
    B, N, M, P, eps, dropout_p, seed, 
    stride_xb, stride_xn, stride_xm,
    stride_yb, stride_ym, stride_yp,
    stride_zb, stride_zn, stride_zp,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr
):
    # Define the ranges for the blocks
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    
    # Compute block indices
    n_block_start = pid_n * BLOCK_SIZE_N
    p_block_start = pid_p * BLOCK_SIZE_P
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_P], dtype=tl.float32)
    
    # Iterate over the shared dimension M
    for m in range(0, M):
        # Load X and Y blocks
        x = tl.load(X_ptr + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_xn + m * stride_xm)
        y = tl.load(Y_ptr + m * stride_ym + (pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)) * stride_yp)
        
        # Perform outer product and accumulate
        acc += x[:, None] * y[None, :]
    
    # Compute RMS normalization
    mean_sq = tl.sum(acc * acc, axis=1) / P
    rms = tl.sqrt(mean_sq + eps)
    normalized = acc / rms[:, None]
    
    # Apply learnable parameter gamma
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE_P))
    normalized *= gamma[None, :]
    
    # Apply GELU activation
    if approximate == 'none':
        gelu = 0.5 * normalized * (1.0 + tl.erf(normalized / tl.sqrt(2.0)))
    elif approximate == 'tanh':
        gelu = 0.5 * normalized * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.14159) * (normalized + 0.044715 * normalized * normalized * normalized)))
    
    # Apply dropout
    if training:
        # Generate random numbers for dropout
        rng = tl.rand(seed)
        mask = rng > dropout_p
        gelu *= mask / (1.0 - dropout_p)
    
    # Store the result
    tl.store(Z_ptr + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_zn + (pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)) * stride_zp, gelu)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Validate input shapes
    assert input1.shape[0] == input2.shape[0], "Batch size must match"
    assert input1.shape[2] == input2.shape[1], "Inner dimensions must match for bmm"
    
    B, N, M = input1.shape
    P = input2.shape[2]
    
    # Allocate output tensor
    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
    
    # Initialize gamma parameter for RMS normalization
    gamma = torch.ones(P, device=input1.device, dtype=input1.dtype)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(N, 128), triton.cdiv(P, 128))
    seed = torch.randint(0, 2**31, (1,), device=input1.device).item() if training else 0
    
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, out, gamma,
        B, N, M, P, eps, dropout_p, seed,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_N=128, BLOCK_SIZE_P=128
    )
    
    return out
