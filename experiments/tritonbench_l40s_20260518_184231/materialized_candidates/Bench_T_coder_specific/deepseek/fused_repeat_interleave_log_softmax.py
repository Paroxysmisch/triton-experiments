import triton
import triton.language as tl

@triton.jit
def fused_repeat_interleave_log_softmax(
    input_ptr,
    repeats_ptr,
    output_ptr,
    n_elements,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Define the index
    pid = tl.program_id()
    block_start = pid * BLOCK_SIZE
    grid_stride = tl.num_programs() * BLOCK_SIZE

    # Repeat interleave
    repeat_buffer = tl.zeros((BLOCK_SIZE,), dtype=input_ptr.dtype)
    for i in range(block_start, n_elements, grid_stride):
        repeat_buffer[i - block_start] = input_ptr[i]

    # Log-softmax
    exp_buffer = tl.exp(repeat_buffer)
    sum_buffer = tl.sum(exp_buffer)
    log_softmax_buffer = tl.log(exp_buffer / sum_buffer)

    # Store the result
    for i in range(block_start, n_elements, grid_stride):
        output_ptr[i] = log_softmax_buffer[i - block_start]
