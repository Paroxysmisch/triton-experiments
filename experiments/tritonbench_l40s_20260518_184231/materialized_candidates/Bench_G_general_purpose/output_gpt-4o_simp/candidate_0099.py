import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    output_ptr,       # Pointer to the output tensor
    input_ptr,        # Pointer to the input tensor
    seq_len_ptr,      # Pointer to the sequence lengths
    num_heads,        # Number of heads
    head_dim,         # Dimension of each head
    batch_stride,     # Stride between batches
    head_stride,      # Stride between heads
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    # Get the program ID
    pid = tl.program_id(0)

    # Calculate batch and head indices
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Load sequence length for the current batch
    seq_len = tl.load(seq_len_ptr + batch_id)

    # Initialize accumulators and logic variables
    acc_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc_sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Process each block of sequence data
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate block start and end indices
        block_start = i
        block_end = min(i + BLOCK_SIZE, seq_len)

        # Load input values and logic sums
        input_offset = batch_id * batch_stride + head_id * head_stride + block_start * head_dim
        input_values = tl.load(input_ptr + input_offset)

        # Compute scaling factors (example: using softmax scaling)
        max_value = tl.max(input_values, axis=0)
        exp_values = tl.exp(input_values - max_value)
        sum_exp_values = tl.sum(exp_values, axis=0)

        # Accumulate results
        acc_value += exp_values * input_values
        acc_sum_exp += sum_exp_values

    # Normalize the accumulated value by the sum of exponentials
    normalized_value = acc_value / acc_sum_exp

    # Store the final output
    output_offset = batch_id * batch_stride + head_id * head_stride
    tl.store(output_ptr + output_offset, normalized_value)

def flash_decode_stage2(
    output, input, seq_len, num_heads, head_dim, BLOCK_SIZE=128
):
    # Get the number of batches from the shape of the input tensor
    num_batches = input.shape[0]

    # Calculate grid size
    grid_size = num_batches * num_heads

    # Launch the Triton kernel
    _fwd_kernel_flash_decode_stage2[grid_size](
        output, input, seq_len, num_heads, head_dim,
        input.stride(0), input.stride(1), BLOCK_SIZE=BLOCK_SIZE
    )
