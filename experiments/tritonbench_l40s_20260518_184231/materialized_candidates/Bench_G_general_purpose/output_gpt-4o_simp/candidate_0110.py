import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr,  # pointer to the input matrix
    lora_ptr,   # pointer to the LoRA weights
    output_ptr, # pointer to the output matrix
    BATCH,      # number of batches
    M,          # number of rows in the input matrix
    N,          # number of columns in the input matrix
    K,          # number of columns in the LoRA matrix
    stride_in_batch, stride_in_m, stride_in_n,  # strides for input matrix
    stride_lora_k, stride_lora_n,               # strides for LoRA weights
    stride_out_batch, stride_out_m,             # strides for output matrix
    ADD_INPUTS: tl.constexpr,                   # whether to add input values
    BLOCK_SIZE: tl.constexpr                    # block size for the computation
):
    # Program IDs
    batch_id = tl.program_id(0)
    row_id = tl.program_id(1)

    # Pointers to the start of the batch and row
    input_ptr = input_ptr + batch_id * stride_in_batch + row_id * stride_in_m
    output_ptr = output_ptr + batch_id * stride_out_batch + row_id * stride_out_m

    # Initialize result
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the LoRA columns
    for k in range(0, K, BLOCK_SIZE):
        # Load input and LoRA blocks
        input_block = tl.load(input_ptr + k * stride_in_n + tl.arange(0, BLOCK_SIZE))
        lora_block = tl.load(lora_ptr + k * stride_lora_k + tl.arange(0, BLOCK_SIZE))

        # Compute the dot product
        result += input_block * lora_block

    # If ADD_INPUTS is set, add existing output values
    if ADD_INPUTS:
        existing_output = tl.load(output_ptr + tl.arange(0, BLOCK_SIZE))
        result += existing_output

    # Store the result
    tl.store(output_ptr + tl.arange(0, BLOCK_SIZE), result)

import torch

def _bgmv_expand(input_matrix, lora_weights, output_matrix, add_inputs=False):
    # Ensure input tensors are on the same device
    assert input_matrix.device == lora_weights.device == output_matrix.device
    device = input_matrix.device

    # Get the dimensions and strides
    BATCH, M, N = input_matrix.shape
    _, _, K = lora_weights.shape

    stride_in_batch, stride_in_m, stride_in_n = input_matrix.stride()
    stride_lora_k, stride_lora_n = lora_weights.stride()[-2:]
    stride_out_batch, stride_out_m = output_matrix.stride()

    # Define block size
    BLOCK_SIZE = 128  # This can be tuned based on the GPU architecture

    # Configure grid size
    grid = (BATCH, M)

    # Launch the kernel
    triton._C.launch(
        _bgmv_expand_kernel,
        grid=grid,
        num_warps=4,  # This can be tuned
        BLOCK_SIZE=BLOCK_SIZE,
        input_ptr=input_matrix.data_ptr(),
        lora_ptr=lora_weights.data_ptr(),
        output_ptr=output_matrix.data_ptr(),
        BATCH=BATCH,
        M=M,
        N=N,
        K=K,
        stride_in_batch=stride_in_batch,
        stride_in_m=stride_in_m,
        stride_in_n=stride_in_n,
        stride_lora_k=stride_lora_k,
        stride_lora_n=stride_lora_n,
        stride_out_batch=stride_out_batch,
        stride_out_m=stride_out_m,
        ADD_INPUTS=add_inputs
    )

# Example usage
input_matrix = torch.randn(10, 128, 256, device='cuda')
lora_weights = torch.randn(10, 256, 64, device='cuda')
output_matrix = torch.zeros(10, 128, 64, device='cuda')

_bgmv_expand(input_matrix, lora_weights, output_matrix, add_inputs=True)
