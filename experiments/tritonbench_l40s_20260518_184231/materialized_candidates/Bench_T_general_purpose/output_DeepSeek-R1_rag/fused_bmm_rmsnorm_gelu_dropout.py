import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    # Pointers to matrices
    input1_ptr, input2_ptr, output_ptr,
    # Matrix dimensions
    B, N, M, P,
    # Strides for input1
    stride_b1, stride_n1, stride_m1,
    # Strides for input2
    stride_b2, stride_m2, stride_p2,
    # Strides for output
    stride_bo, stride_no, stride_po,
    # RMSNorm parameters
    eps,
    # Dropout parameters
    dropout_p, training,
    # GELU approximation
    APPROXIMATE: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    # Batch and Row indices
    b_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    if b_idx >= B or n_idx >= N:
        return
    
    # Compute the offset for the current row in input1
    input1_row_ptr = input1_ptr + b_idx * stride_b1 + n_idx * stride_n1
    
    # Accumulator for the output row
    acc = tl.zeros((BLOCK_SIZE_P,), dtype=tl.float32)
    
    # Compute the matrix multiplication for the current row
    for m_idx in range(0, M, BLOCK_SIZE_M):
        m_offs = m_idx + tl.arange(0, BLOCK_SIZE_M)
        a_mask = m_offs < M
        a = tl.load(input1_row_ptr + m_offs * stride_m1, mask=a_mask, other=0.0)
        
        for p_idx in range(0, P, BLOCK_SIZE_P):
            p_offs = p_idx + tl.arange(0, BLOCK_SIZE_P)
            b_mask = (m_offs[:, None] < M) & (p_offs[None, :] < P)
            b_ptr = input2_ptr + b_idx * stride_b2 + m_offs[:, None] * stride_m2 + p_offs[None, :] * stride_p2
            b = tl.load(b_ptr, mask=b_mask, other=0.0)
            
            a_fp16 = a.to(tl.float16)
            b_fp16 = b.to(tl.float16)
            acc_inc = tl.sum(a_fp16[:, None] * b_fp16, axis=0)
            acc += acc_inc.to(tl.float32)
    
    # Compute RMSNorm
    sum_sq = tl.sum(acc * acc)
    mean_sq = sum_sq / P
    rms_scale = tl.math.rsqrt(mean_sq + eps)
    normalized = acc * rms_scale
    
    # Apply GELU
    if APPROXIMATE == 'tanh':
        gelu = 0.5 * normalized * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.math.pi) * (normalized + 0.044715 * normalized * normalized * normalized)))
    else:
        gelu = 0.5 * normalized * (1.0 + tl.erf(normalized / tl.sqrt(2.0)))
    
    # Apply dropout
    if training:
        seed = tl.randint(0, 2**32, ())
        mask = tl.rand(seed, (BLOCK_SIZE_P,)) > dropout_p
        dropout_scale = 1.0 / (1.0 - dropout_p)
        output = gelu * mask * dropout_scale
    else:
        output = gelu
    
    # Store the result
    for p_idx in range(0, P, BLOCK_SIZE_P):
        p_offs = p_idx + tl.arange(0, BLOCK_SIZE_P)
        mask = p_offs < P
        out_ptr = output_ptr + b_idx * stride_bo + n_idx * stride_no + p_offs * stride_po
        tl.store(out_ptr, output[p_offs], mask=mask)

def fused_bmm_rmsnorm_gelu_dropout(
    input1: torch.Tensor,
    input2: torch.Tensor,
    normalized_shape,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
    training: bool = True,
    approximate: str = 'none',
    *, out: torch.Tensor = None
) -> torch.Tensor:
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3, "Inputs must be 3D tensors"
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Input dimensions mismatch"
    
    # Check normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    output_shape = (B, N, P)
    assert tuple(normalized_shape) == (output_shape[-1],), "normalized_shape must match last dimension"
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(output_shape, device=input1.device, dtype=input1.dtype)
    else:
        assert out.shape == output_shape, "Output shape mismatch"
    
    # Define kernel grid and block sizes
    grid = (B, N)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_P = 128  # Tune based on hardware
    
    # Launch kernel
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        eps, dropout_p, training,
        'tanh' if approximate == 'tanh' else 'none',
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_P,
    )
    
    return out
