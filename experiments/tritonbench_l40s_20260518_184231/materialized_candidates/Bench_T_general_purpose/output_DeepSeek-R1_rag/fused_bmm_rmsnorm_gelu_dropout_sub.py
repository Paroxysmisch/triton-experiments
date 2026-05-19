import torch
import triton
import triton.language as tl

@triton.jit
def _fused_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    B, N, M, P,
    input1_b_stride, input1_n_stride, input1_m_stride,
    input2_b_stride, input2_m_stride, input2_p_stride,
    other_b_stride, other_n_stride, other_p_stride,
    output_b_stride, output_n_stride, output_p_stride,
    eps, dropout_p, training,
    APPROXIMATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (B * N)
    n = (pid // P) % N
    p = pid % P
    
    # Compute batch matrix multiplication
    z = 0.0
    for m in range(M):
        input1_offset = b * input1_b_stride + n * input1_n_stride + m * input1_m_stride
        input2_offset = b * input2_b_stride + m * input2_m_stride + p * input2_p_stride
        a = tl.load(input1_ptr + input1_offset)
        b_val = tl.load(input2_ptr + input2_offset)
        z += a * b_val

    # RMS normalization
    mean_sq = tl.sum(z * z) / P
    rms = tl.sqrt(mean_sq + eps)
    z_norm = z / rms

    # GELU activation
    if APPROXIMATE == 'tanh':
        gelu = 0.5 * z_norm * (1 + tl.tanh(tl.sqrt(2 / tl.math.pi) * (z_norm + 0.044715 * tl.pow(z_norm, 3))))
    else:
        gelu = 0.5 * z_norm * (1 + tl.erf(z_norm / tl.math.sqrt(2.0)))

    # Dropout
    if training:
        keep_prob = 1.0 - dropout_p
        mask = tl.rand(tl.load(input1_ptr + input1_offset).to(tl.float32)) < keep_prob
        gelu = tl.where(mask, gelu / keep_prob, 0.0)

    # Subtract other tensor
    other_offset = b * other_b_stride + n * other_n_stride + p * other_p_stride
    other_val = tl.load(other_ptr + other_offset)
    result = gelu - other_val

    # Store result
    output_offset = b * output_b_stride + n * output_n_stride + p * output_p_stride
    tl.store(output_ptr + output_offset, result)

def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1: torch.Tensor, input2: torch.Tensor, other: torch.Tensor,
    normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None
):
    # Validate input shapes
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Invalid input shapes for batch matrix multiplication"
    
    # Create output tensor
    output_shape = (B, N, P)
    if out is None:
        out = torch.empty(output_shape, device=input1.device, dtype=input1.dtype)
    
    # Reshape inputs for kernel
    input1_reshaped = input1.view(B * N, M)
    input2_reshaped = input2.view(B * M, P)
    other_reshaped = other.broadcast_to(output_shape).contiguous()
    
    # Determine kernel parameters
    grid = (B * N * P,)
    BLOCK_SIZE = triton.next_power_of_2(max(M, P))
    DTYPE = tl.float16 if input1.dtype == torch.float16 else tl.float32
    
    # Launch kernel
    _fused_kernel[grid](
        input1_reshaped, input2_reshaped, other_reshaped, out,
        B, N, M, P,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other_reshaped.stride(0), other_reshaped.stride(1), other_reshaped.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        eps, dropout_p, training,
        APPROXIMATE=approximate,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=DTYPE,
    )
    return out
