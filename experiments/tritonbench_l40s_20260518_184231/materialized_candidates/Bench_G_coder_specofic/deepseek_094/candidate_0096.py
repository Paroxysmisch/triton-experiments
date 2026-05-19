import triton
import triton.language as tl

BLOCK_SIZE = 32

@triton.jit
def _rms_norm_fwd_fused(x_ptr, y_ptr, weight_ptr, n, rstd_ptr, mask, out_n):
    # Define the indices
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    block_start = col * BLOCK_SIZE

    # Load the data
    x = tl.load(x_ptr + block_start + row)
    weight = tl.load(weight_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    # Compute the variance
    var = tl.sum(x * x) / n
    var = tl.program_scope(var, lambda var: tl.where(col == 0, var, var.previous))

    # Compute the normalized values
    y = (x / tl.sqrt(var)) * rstd * weight

    # Store the result
    tl.store(y_ptr + block_start + row, y)

def rmsnorm_forward(x, weight, y, rstd):
    # Compute the reciprocal standard deviation
    rstd = 1 / tl.sqrt(rstd)

    # Compute the number of columns
    n = x.num_elements()

    # Compute the number of warps
    num_warps = triton.next_power_of_2(n) // BLOCK_SIZE

    # Compute the mask
    mask = triton.mask(n, triton.program_id())

    # Compute the output size
    out_n = n * num_warps

    # Launch the kernel
    _rms_norm_fwd_fused[num_warps, BLOCK_SIZE](x.ptr, y.ptr, weight.ptr, n, rstd.ptr, mask, out_n)

    return y
