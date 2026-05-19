import triton
import triton.language as tl

@triton.jit
def bitwise_and_binomial_kernel(
    output_ptr,
    input_ptr,
    other_ptr,
    total_count_ptr,
    n,
    dtype):
    
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n, BLOCK_SIZE)

    if pid >= grid_size:
        return

    # Load data
    input_val = tl.load(input_ptr + pid * BLOCK_SIZE)
    other_val = tl.load(other_ptr + pid * BLOCK_SIZE)
    total_count_val = tl.load(total_count_ptr + pid * BLOCK_SIZE)

    # Bitwise AND
    and_result = input_val & other_val

    # Convert and_result to int32
    and_result_int = tl.convert_element_type(and_result, tl.int32)

    # Sample from Binomial distribution
    if dtype == tl.float32:
        binom_sample = tl.random.binomial(total_count_val, and_result_int / 255.0)
    elif dtype == tl.float64:
        binom_sample = tl.random.binomial(total_count_val, and_result_int / 255.0)

    # Store the result
    tl.store(output_ptr + pid * BLOCK_SIZE, binom_sample)

BLOCK_SIZE = 1024
