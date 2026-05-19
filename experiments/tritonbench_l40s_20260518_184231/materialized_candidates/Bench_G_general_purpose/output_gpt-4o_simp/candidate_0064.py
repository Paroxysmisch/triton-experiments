import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,       # Pointer to the input tensor
    weight_ptr,      # Pointer to the LoRA weight tensor
    output_ptr,      # Pointer to the output tensor
    batch_size,      # Number of batches
    hidden_size,     # Size of the hidden layer
    output_size,     # Size of the output layer
    BLOCK_SIZE: tl.constexpr,  # Block size for tiling
    ACCUMULATE: tl.constexpr   # Whether to accumulate with existing output
):
    # Calculate the program index
    pid = tl.program_id(axis=0)

    # Determine the block start for the batch and hidden dimensions
    batch_start = pid // (hidden_size // BLOCK_SIZE)
    hidden_start = (pid % (hidden_size // BLOCK_SIZE)) * BLOCK_SIZE

    # Create pointers for the input and weight blocks
    input_block_ptr = input_ptr + batch_start * hidden_size + hidden_start
    weight_block_ptr = weight_ptr + hidden_start * output_size

    # Create a pointer for the output block
    output_block_ptr = output_ptr + batch_start * output_size

    # Load input and weight blocks
    input_block = tl.load(input_block_ptr, mask=hidden_start + tl.arange(0, BLOCK_SIZE) < hidden_size)
    weight_block = tl.load(weight_block_ptr)

    # Perform matrix-vector multiplication
    result = tl.dot(input_block, weight_block)

    # Accumulate with existing output if needed
    if ACCUMULATE:
        existing_output = tl.load(output_block_ptr)
        result += existing_output

    # Store the result
    tl.store(output_block_ptr, result)

### Wrapper Function: `_bgmv_expand_slice`

The wrapper function sets up the kernel with the appropriate parameters and launches it.
