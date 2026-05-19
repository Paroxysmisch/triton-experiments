import triton
import triton.language as tl

# Triton kernel for seeded dropout
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # Write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function for seeded dropout
def seeded_dropout(x, p, seed=0, BLOCK_SIZE=64):
    # Ensure x is contiguous
    x = x.contiguous()

    # Calculate grid size
    n_elements = x.numel()
    grid = lambda meta: triton.TensorDevice(meta, tl.nn.scale_symbol(n_elements, BLOCK_SIZE))

    # Create an output tensor of same shape and type as x
    output = x.new().resize_as_(x)

    # Launch Triton kernel
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE)

    return output
