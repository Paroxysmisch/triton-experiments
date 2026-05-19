import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr,  # Pointer to the input matrix
    lora_weights_ptr,  # Pointer to the LoRA weights
    output_ptr,  # Pointer to the output matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    K,  # Number of columns in the LoRA weights
    stride_input_m,  # Stride of the input matrix in the M dimension
    stride_input_n,  # Stride of the input matrix in the N dimension
    stride_lora_k,  # Stride of the LoRA weights in the K dimension
    stride_lora_n,  # Stride of the LoRA weights in the N dimension
    stride_output_m,  # Stride of the output matrix in the M dimension
    stride_output_k,  # Stride of the output matrix in the K dimension
    ADD_INPUTS: tl.constexpr,  # Flag to add existing output values
    BLOCK_SIZE_M: tl.constexpr,  # Block size for the M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for the N dimension
    BLOCK_SIZE_K: tl.constexpr  # Block size for the K dimension
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = (pid - first_pid_m * num_pid_n) // BLOCK_SIZE_N
    pid_m = first_pid_m + tl.arange(0, BLOCK_SIZE_M)
    pid_n = first_pid_n + tl.arange(0, BLOCK_SIZE_N)
    pid_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute the offsets for the input, LoRA weights, and output matrices
    offsets_input = pid_m[:, None] * stride_input_m + pid_n[None, :] * stride_input_n
    offsets_lora = pid_k[:, None] * stride_lora_k + pid_n[None, :] * stride_lora_n
    offsets_output = pid_m[:, None] * stride_output_m + pid_k[None, :] * stride_output_k

    # Load the input and LoRA weights
    input_block = tl.load(input_ptr + offsets_input, mask=pid_m[:, None] < M, other=0.0)
    lora_weights_block = tl.load(lora_weights_ptr + offsets_lora, mask=pid_k[:, None] < K, other=0.0)

    # Compute the matrix-vector product
    output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    output_block += tl.dot(input_block, lora_weights_block)

    # Add existing output values if ADD_INPUTS is set
    if ADD_INPUTS:
        existing_output_block = tl.load(output_ptr + offsets_output, mask=pid_m[:, None] < M, other=0.0)
        output_block += existing_output_block

    # Store the result in the output matrix
    tl.store(output_ptr + offsets_output, output_block, mask=pid_m[:, None] < M)

import torch
import triton
import triton.language as tl

def _bgmv_expand(input, lora_weights, output, add_inputs=False, dtype=None):
    # Check input tensor properties
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert lora_weights.dim() == 2, "LoRA weights must be a 2D tensor"
    assert output.dim() == 2, "Output must be a 2D tensor"
    assert input.size(1) == lora_weights.size(1), "Input and LoRA weights must have the same number of columns"
    assert input.size(0) == output.size(0), "Input and output must have the same number of rows"
    assert lora_weights.size(0) == output.size(1), "LoRA weights and output must have the same number of columns"

    # Get the problem size and strides
    M, N = input.size()
    K = lora_weights.size(0)
    stride_input_m, stride_input_n = input.stride()
    stride_lora_k, stride_lora_n = lora_weights.stride()
    stride_output_m, stride_output_k = output.stride()

    # Cast the input and output tensors if necessary
    if dtype is not None:
        input = input.to(dtype)
        lora_weights = lora_weights.to(dtype)
        output = output.to(dtype)

    # Configure the grid
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the Triton kernel
    _bgmv_expand_kernel[grid](
        input, lora_weights, output, M, N, K, stride_input_m, stride_input_n, stride_lora_k, stride_lora_n, stride_output_m, stride_output_k, add_inputs, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return output

import torch

# Create input, LoRA weights, and output tensors
input = torch.randn(128, 64, device='cuda')
lora_weights = torch.randn(32, 64, device='cuda')
output = torch.zeros(128, 32, device='cuda')

# Perform the batched generalized matrix-vector multiplication
output = _bgmv_expand(input, lora_weights, output, add_inputs=True, dtype=torch.float16)

print(output)
