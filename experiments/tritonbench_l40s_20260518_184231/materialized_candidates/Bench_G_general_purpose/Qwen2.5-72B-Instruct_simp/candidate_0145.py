import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    int8_input_ptr,  # Pointer to the int8 input matrix
    float_output_ptr,  # Pointer to the float output matrix
    scale_factor,  # Scale factor for dequantization
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    stride_im,  # Stride for rows in the int8 input matrix
    stride_in,  # Stride for columns in the int8 input matrix
    stride_fm,  # Stride for rows in the float output matrix
    stride_fn,  # Stride for columns in the float output matrix
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_program = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N

    offsets_m = block_offset_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_offset_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    for i in range(0, BLOCK_SIZE_M, 32):
        for j in range(0, BLOCK_SIZE_N, 32):
            offsets = offsets_m[i:i+32] * stride_im + offsets_n[j:j+32] * stride_in
            int8_values = tl.load(int8_input_ptr + offsets, mask=mask_m[i:i+32] & mask_n[j:j+32], other=0)

            float_values = int8_values * scale_factor

            offsets = offsets_m[i:i+32] * stride_fm + offsets_n[j:j+32] * stride_fn
            tl.store(float_output_ptr + offsets, float_values, mask=mask_m[i:i+32] & mask_n[j:j+32])

import torch
import triton
import triton.language as tl

def matmul_dequantize_int8(a: torch.Tensor, int8_b: torch.Tensor, scale_factor: float) -> torch.Tensor:
    # Ensure input tensors are on the same device
    device = a.device
    assert int8_b.device == device, "Input tensors must be on the same device"

    # Ensure input tensors have compatible dimensions
    M, K = a.shape
    K, N = int8_b.shape
    assert K == int8_b.shape[0], "Matrix dimensions are not compatible for multiplication"

    # Allocate memory for the dequantized matrix
    float_b = torch.empty((K, N), dtype=torch.float32, device=device)

    # Launch the dequantize kernel
    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    dequantize_kernel[grid](int8_b, float_b, scale_factor, K, N, int8_b.stride(0), int8_b.stride(1), float_b.stride(0), float_b.stride(1), BLOCK_SIZE_M=16, BLOCK_SIZE_N=16)

    # Perform the matrix multiplication
    c = torch.matmul(a, float_b)

    return c
