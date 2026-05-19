import triton
import triton.language as tl

# Define the Triton kernel for Swiglu forward computation
@triton.jit
def _swiglu_fwd_kernel(X, Y, OUT, stride_xm, stride_ym, stride_om, N, **meta):
    # Obtain the program ID for parallel execution
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this program instance
    offs = pid * meta['BLOCK_SIZE_N'] + tl.arange(0, meta['BLOCK_SIZE_N'])

    # Ensure we don't read/write out of bounds
    mask = offs < N

    # Load elements from X and Y
    x = tl.load(X + offs * stride_xm, mask=mask, other=0.0)
    y = tl.load(Y + offs * stride_ym, mask=mask, other=0.0)

    # Compute the Swish activation: x * sigmoid(x)
    swish_x = x * tl.sigmoid(x)

    # Compute the output: swish_x * y
    out = swish_x * y

    # Store the result in the output matrix
    tl.store(OUT + offs * stride_om, out, mask=mask)

# Wrapper function to set up and invoke the Triton kernel
def _swiglu_fwd(x, y):
    # Ensure the input tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Check that the dimensions of x and y match
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Get the number of elements in the input tensors
    N = x.numel()

    # Create an output tensor with the same shape
    out = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE_N = 1024

    # Calculate the number of blocks needed
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the Triton kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        x.stride(0), y.stride(0), out.stride(0),
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return out
