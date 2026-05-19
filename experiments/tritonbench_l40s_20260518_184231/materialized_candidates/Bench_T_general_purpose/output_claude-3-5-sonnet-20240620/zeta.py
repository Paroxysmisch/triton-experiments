import triton
import triton.language as tl

@triton.jit
def _hurwitz_zeta_kernel(
    x_ptr,      # pointer to x values
    q_ptr,      # pointer to q values
    out_ptr,    # pointer to output
    n_elements, # number of elements
    BLOCK_SIZE: tl.constexpr,
    N_TERMS: tl.constexpr,  # number of terms in the series approximation
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start index
    block_start = pid * BLOCK_SIZE
    
    # Load x and q values
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)
    
    # Initialize sum
    result = tl.zeros_like(x)
    
    # Compute the series sum
    for k in range(N_TERMS):
        k_float = tl.float32(k)
        denominator = tl.pow(k_float + q, x)
        result += 1.0 / denominator
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Wrapper function
def zeta(input, other, *, out=None):
    """
    Computes the Hurwitz zeta function elementwise.
    
    Args:
        input (Tensor): the input tensor corresponding to x
        other (Tensor): the input tensor corresponding to q
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: Result of the Hurwitz zeta function
    """
    # Input validation
    assert input.shape == other.shape, "Input tensors must have the same shape"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    # Get tensor properties
    n_elements = input.numel()
    
    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Number of terms in the series approximation
    N_TERMS = 1000  # This can be adjusted based on desired accuracy
    
    # Launch kernel
    _hurwitz_zeta_kernel[grid](
        input.contiguous().data_ptr(),
        other.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        N_TERMS=N_TERMS,
    )
    
    return out
