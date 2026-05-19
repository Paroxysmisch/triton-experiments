import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,  # Pointer to the input tensor
    lora_weight_ptr,  # Pointer to the LoRA weight tensor
    output_ptr,  # Pointer to the output tensor
    batch_size,  # Number of batches
    input_size,  # Size of the input tensor
    output_size,  # Size of the output tensor
    lora_rank,  # Rank of the LoRA weight tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
    ACCUMULATE: tl.constexpr  # Flag to accumulate with existing output
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (output_size // BLOCK_SIZE)
    output_offset = (pid % (output_size // BLOCK_SIZE)) * BLOCK_SIZE

    # Compute the input and output offsets
    input_offset = batch_id * input_size
    output_offset += batch_id * output_size

    # Load the input vector
    input_vec = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < input_size, other=0.0)

    # Load the LoRA weight matrix
    lora_weight = tl.load(lora_weight_ptr + batch_id * input_size * lora_rank, mask=tl.arange(0, input_size * lora_rank) < input_size * lora_rank, other=0.0)
    lora_weight = tl.reshape(lora_weight, (input_size, lora_rank))

    # Perform the matrix-vector multiplication
    output_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(lora_rank):
        lora_weight_col = lora_weight[:, i]
        output_vec += lora_weight_col * input_vec

    # Accumulate with existing output if required
    if ACCUMULATE:
        existing_output = tl.load(output_ptr + output_offset, mask=tl.arange(0, BLOCK_SIZE) < output_size, other=0.0)
        output_vec += existing_output

    # Store the result
    tl.store(output_ptr + output_offset, output_vec, mask=tl.arange(0, BLOCK_SIZE) < output_size)

import torch

def _bgmv_expand_slice(input_tensor, lora_weight_tensor, output_tensor, batch_size, input_size, output_size, lora_rank, accumulate=False):
    # Convert tensors to Triton pointers
    input_ptr = input_tensor.data_ptr()
    lora_weight_ptr = lora_weight_tensor.data_ptr()
    output_ptr = output_tensor.data_ptr()

    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = (batch_size * (output_size // BLOCK_SIZE),)

    # Launch the kernel
    _bgmv_expand_slice_kernel[grid](
        input_ptr,
        lora_weight_ptr,
        output_ptr,
        batch_size,
        input_size,
        output_size,
        lora_rank,
        BLOCK_SIZE,
        accumulate
    )

# Example usage
if __name__ == "__main__":
    batch_size = 32
    input_size = 1024
    output_size = 512
    lora_rank = 64

    # Create input, LoRA weight, and output tensors
    input_tensor = torch.randn(batch_size, input_size, device='cuda')
    lora_weight_tensor = torch.randn(batch_size, input_size, lora_rank, device='cuda')
    output_tensor = torch.zeros(batch_size, output_size, device='cuda')

    # Perform the GroupGEMV operation
    _bgmv_expand_slice(input_tensor, lora_weight_tensor, output_tensor, batch_size, input_size, output_size, lora_rank, accumulate=True)

    print(output_tensor)
