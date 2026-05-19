import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({'num_warps': 1}),
    triton.Config({'num_warps': 2}),
    triton.Config({'num_warps': 4}),
    triton.Config({'num_warps': 8}),
    triton.Config({'num_warps': 16}),
    triton.Config({'num_warps': 32}),
], key=['D'])
@triton.jit
def logsumexp_fwd_kernel(x_ptr, z_ptr, N, D, HAS_SCALE, scale_ptr, **meta):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Calculate the start of the block for this program
    start_d = pid_d * meta['BLOCK_SIZE_D']
    
    # Create offsets for the current block
    offsets_d = start_d + tl.arange(0, meta['BLOCK_SIZE_D'])
    offsets_n = pid_n

    # Create mask to handle out-of-bounds accesses
    mask = offsets_d < D

    # Load input data into a block
    b_x = tl.load(x_ptr + offsets_n * D + offsets_d, mask=mask, other=-float('inf'))

    # Optionally scale the input
    if HAS_SCALE:
        scale = tl.load(scale_ptr + offsets_d, mask=mask)
        b_x *= scale

    # Compute the maximum value along the block dimension
    b_m = tl.max(b_x, axis=0)

    # Compute the exponential and sum it
    b_e = tl.exp(b_x - b_m)
    b_s = tl.sum(b_e, axis=0)

    # Compute log-sum-exp
    b_lse = tl.log(b_s) + b_m

    # Store the result
    tl.store(z_ptr + offsets_n * D + offsets_d, b_lse, mask=mask)

# Wrapper function
def logsumexp_fwd(x, has_scale=False, scale=None, output_dtype=None):
    # Reshape input tensor
    N, D = x.shape
    BLOCK_SIZE_D = 128  # Example block size
    ND = (D + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D

    # Create output tensor
    z = torch.empty_like(x)

    # Launch the kernel
    logsumexp_fwd_kernel[(N, ND)](
        x, z, N, D, has_scale, scale,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )

    # Reduce along the last dimension to produce the final result
    result = z.view(N, ND, BLOCK_SIZE_D).sum(dim=-1)

    # Cast to desired output type if specified
    if output_dtype is not None:
        result = result.to(output_dtype)

    return result
