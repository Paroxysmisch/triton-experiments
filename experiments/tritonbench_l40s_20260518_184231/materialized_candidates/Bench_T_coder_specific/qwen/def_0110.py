import triton
import triton.language as tl

@triton.jit
def exp_kernel(
    x_ptr,
    exp_x_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    coords = pid * block_size + tl.arange(0, block_size)
    valid_mask = coords < n_elements

    # Apply exponential function
    x_val = tl.load(x_ptr + coords, mask=valid_mask)
    exp_x_val = tl.exp(x_val)

    # Reduce to compute sum of exponentials
    exp_sum = tl.sum(exp_x_val, axis=0)

    # Store the result
    tl.store(exp_x_ptr + pid, exp_sum, mask=pid == 0)
