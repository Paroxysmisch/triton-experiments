import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    # Pointers to matrices
    output_ptr, input1_ptr, input2_ptr, other_ptr, weights_ptr,
    # Matrix dimensions
    batch_size, N, M, P,
    # Strides for the different tensors
    input1_batch_stride, input1_row_stride,
    input2_batch_stride, input2_row_stride,
    other_batch_stride, other_row_stride,
    output_batch_stride, output_row_stride,
    # Additional parameters
    eps, dropout_p, seed,
    # Whether we're training (for dropout)
    training: tl.constexpr,
    # GELU approximation type
    approximate: tl.constexpr,
    # Block sizes for the GEMM
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel combining batch matrix multiplication, RMS normalization,
    GELU activation, dropout, and subtraction.
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_within_batch = pid % num_pid_in_batch
    
    # Get the block indices
    block_m = (pid_within_batch // num_pid_n) * BLOCK_SIZE_M
    block_n = (pid_within_batch % num_pid_n) * BLOCK_SIZE_N

    # Initialize pointers to current batch
    input1_batch_ptr = input1_ptr + batch_id * input1_batch_stride
    input2_batch_ptr = input2_ptr + batch_id * input2_batch_stride
    other_batch_ptr = other_ptr + batch_id * other_batch_stride
    output_batch_ptr = output_ptr + batch_id * output_batch_stride

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Batch matrix multiplication
    for k in range(0, M, BLOCK_SIZE_K):
        # Pointers to rows of input matrices for this block
        a_ptr = input1_batch_ptr + block_m * input1_row_stride + k
        b_ptr = input2_batch_ptr + k * input2_row_stride + block_n

        # Load inputs
        a = tl.load(a_ptr, mask=block_m + tl.arange(0, BLOCK_SIZE_M) < N)
        b = tl.load(b_ptr, mask=block_n + tl.arange(0, BLOCK_SIZE_N) < P)

        # Compute matrix multiplication for this block
        acc += tl.dot(a, b)

    # RMS Normalization
    square_sum = tl.sum(acc * acc, axis=1) / P
    rms = tl.sqrt(square_sum + eps)
    normalized = acc / rms[:, None]

    # GELU activation
    if approximate == 'tanh':
        # Tanh approximation
        sqrt_2_over_pi = 0.7978845608028654
        coef = 0.044715
        x = normalized
        x3 = x * x * x
        inner = sqrt_2_over_pi * (x + coef * x3)
        gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    else:
        # Exact GELU
        x = normalized
        cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        gelu = x * cdf

    # Dropout
    if training:
        rand = tl.rand(seed, gelu.shape)
        dropout_mask = rand > dropout_p
        gelu = tl.where(dropout_mask, gelu / (1.0 - dropout_p), 0.0)

    # Load other tensor for subtraction
    other_val = tl.load(other_batch_ptr + block_m * other_row_stride + block_n)
    
    # Subtract and store result
    output = gelu - other_val
    output_ptr = output_batch_ptr + block_m * output_row_stride + block_n
    tl.store(output_ptr, output)

def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1, input2, other, normalized_shape, 
    dropout_p=0.5, training=True, 
    approximate='none', eps=1e-5, *, out=None
):
    """
    Fused operation combining batch matrix multiplication, RMS normalization,
    GELU activation, dropout, and subtraction.
    """
    # Input validation
    assert input1.dim() == 3 and input2.dim() == 3, "Input tensors must be 3D"
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Batch size and inner dimensions must match"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
    
    # Compute grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (B * triton.cdiv(N, BLOCK_SIZE_M) * triton.cdiv(P, BLOCK_SIZE_N),)
    
    # Generate random seed for dropout
    seed = torch.randint(0, 2**31-1, (1,)).item() if training else 0
    
    # Create weights for RMS norm (all ones)
    weights = torch.ones(normalized_shape, device=input1.device, dtype=input1.dtype)
    
    # Launch kernel
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        out, input1, input2, other, weights,
        B, N, M, P,
        input1.stride(0), input1.stride(1),
        input2.stride(0), input2.stride(1),
        other.stride(0), other.stride(1),
        out.stride(0), out.stride(1),
        eps, dropout_p, seed,
        training=training,
        approximate=approximate,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out
