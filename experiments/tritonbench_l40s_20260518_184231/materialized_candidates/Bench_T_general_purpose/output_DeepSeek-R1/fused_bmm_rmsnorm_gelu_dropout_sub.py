import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    B, N, M, P,
    eps, dropout_p, training, approximate,
    stride_in1_b, stride_in1_n, stride_in1_m,
    stride_in2_b, stride_in2_m, stride_in2_p,
    stride_other_b, stride_other_n, stride_other_p,
    stride_out_b, stride_out_n, stride_out_p,
    SEED: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    
    pid_p = tl.program_id(2)
    num_p_pid = tl.cdiv(P, BLOCK_SIZE_P)
    
    sum_squares = 0.0
    # First pass: Compute sum of squares for RMSNorm
    for p_idx in range(pid_p * BLOCK_SIZE_P, (pid_p + 1) * BLOCK_SIZE_P):
        if p_idx >= P:
            break
        z = 0.0
        for m in range(M):
            off_in1 = b * stride_in1_b + n * stride_in1_n + m * stride_in1_m
            off_in2 = b * stride_in2_b + m * stride_in2_m + p_idx * stride_in2_p
            a = tl.load(input1_ptr + off_in1)
            b_val = tl.load(input2_ptr + off_in2)
            z += a * b_val
        sum_squares += z * z
    sum_squares = tl.sum(sum_squares, axis=0) / num_p_pid
    
    mean_squares = sum_squares / P
    rms = tl.math.rsqrt(mean_squares + eps)
    
    # Second pass: Compute Z, apply RMSNorm, GELU, dropout, and subtract
    for p_idx in range(pid_p * BLOCK_SIZE_P, (pid_p + 1) * BLOCK_SIZE_P):
        if p_idx >= P:
            break
        z = 0.0
        for m in range(M):
            off_in1 = b * stride_in1_b + n * stride_in1_n + m * stride_in1_m
            off_in2 = b * stride_in2_b + m * stride_in2_m + p_idx * stride_in2_p
            a = tl.load(input1_ptr + off_in1)
            b_val = tl.load(input2_ptr + off_in2)
            z += a * b_val
        
        z_norm = z * rms
        
        if approximate == 'none':
            gelu = 0.5 * z_norm * (1.0 + tl.math.erf(z_norm / tl.math.sqrt(2.0)))
        else:
            cdf = 0.5 * (1.0 + tl.math.tanh(tl.math.sqrt(2.0 / tl.math.pi) * (z_norm + 0.044715 * (z_norm ** 3))))
            gelu = z_norm * cdf
        
        if training:
            philox_seed = SEED
            philox_offset = b * N * P + n * P + p_idx
            rand = tl.rand(philox_seed, philox_offset)
            keep = rand > dropout_p
            dropout_factor = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0
            d = gelu * tl.where(keep, 1.0, 0.0) * dropout_factor
        else:
            d = gelu
        
        off_other = b * stride_other_b + n * stride_other_n + p_idx * stride_other_p
        other_val = tl.load(other_ptr + off_other)
        y = d - other_val
        
        off_out = b * stride_out_b + n * stride_out_n + p_idx * stride_out_p
        tl.store(output_ptr + off_out, y)

def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1: torch.Tensor,
    input2: torch.Tensor,
    other: torch.Tensor,
    normalized_shape,
    dropout_p: float = 0.5,
    training: bool = True,
    approximate: str = 'none',
    eps: float = 1e-5,
    out: torch.Tensor = None,
) -> torch.Tensor:
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Input1 and Input2 must be compatible for bmm"
    
    output_shape = (B, N, P)
    if out is None:
        out = torch.empty(output_shape, dtype=input1.dtype, device=input1.device)
    
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    assert list(normalized_shape) == [P], "normalized_shape must match last dimension P"
    
    other_expanded = other.broadcast_to(output_shape).contiguous()
    
    stride_in1_b, stride_in1_n, stride_in1_m = input1.stride()
    stride_in2_b, stride_in2_m, stride_in2_p = input2.stride()
    stride_other_b, stride_other_n, stride_other_p = other_expanded.stride()
    stride_out_b, stride_out_n, stride_out_p = out.stride()
    
    BLOCK_SIZE_P = 128
    grid = (B, N, triton.cdiv(P, BLOCK_SIZE_P))
    
    seed = 0
    if training:
        seed = torch.randint(0, 2**32, (1,), device='cuda').item()
    
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1, input2, other_expanded, out,
        B, N, M, P,
        eps, dropout_p, training, approximate,
        stride_in1_b, stride_in1_n, stride_in1_m,
        stride_in2_b, stride_in2_m, stride_in2_p,
        stride_other_b, stride_other_n, stride_other_p,
        stride_out_b, stride_out_n, stride_out_p,
        SEED=seed,
        BLOCK_SIZE_P=BLOCK_SIZE_P,
    )
    
    return out
