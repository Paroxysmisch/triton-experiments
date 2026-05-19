import triton
import triton.language as tl
import torch
import math

@triton.jit
def chebyshev_kernel(
    input_ptr,  # Pointer to input tensor
    n_ptr,      # Pointer to n tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values and n
    x = tl.load(input_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask)

    # Initialize result
    result = tl.where(n == 0, 1.0, x)
    
    # Compute using recursive formula for n < 6 or |x| > 1
    use_recursive = tl.where((n < 6) | (tl.abs(x) > 1.0), 1.0, 0.0)
    
    # Recursive implementation
    def recursive_chebyshev(x, n):
        t_prev = 1.0  # T_0(x)
        t_curr = x    # T_1(x)
        
        for i in range(2, n + 1):
            t_next = 2.0 * x * t_curr - t_prev
            t_prev = t_curr
            t_curr = t_next
        
        return tl.where(n == 0, t_prev, t_curr)
    
    # Trigonometric implementation
    def trig_chebyshev(x, n):
        theta = tl.arccos(x)
        return tl.cos(n * theta)
    
    # Choose implementation based on conditions
    result = tl.where(
        use_recursive == 1.0,
        recursive_chebyshev(x, n),
        trig_chebyshev(x, n)
    )
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def chebyshev_polynomial_t(input: torch.Tensor, n: torch.Tensor, *, out=None) -> torch.Tensor:
    """
    Computes the Chebyshev polynomial of the first kind T_n(input).
    
    Args:
        input (Tensor): Input tensor
        n (Tensor): Degree of the polynomial
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor: Result of Chebyshev polynomial evaluation
    """
    # Input validation
    assert input.shape == n.shape, "Input and n must have the same shape"
    assert input.is_cuda and n.is_cuda, "Input and n must be CUDA tensors"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Grid and block size configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    chebyshev_kernel[grid](
        input.contiguous().data_ptr(),
        n.contiguous().data_ptr(),
        out.data_ptr(),
        input.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
