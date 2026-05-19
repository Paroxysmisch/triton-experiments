import torch
import triton
import triton.language as tl
import math

@triton.jit
def fft_kernel(
    input_ptr,
    output_ptr,
    stride,
    n_elements,
    signal_size,
    norm_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing FFT.
    This implements the Cooley-Tukey FFT algorithm for power-of-2 sizes.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets * stride, mask=mask)
    
    # Bit reversal permutation
    x = tl.where(mask, x, 0.0)
    
    # FFT computation using butterfly operations
    for stage in range(tl.log2(signal_size)):
        distance = 1 << stage
        for k in range(0, signal_size, 2 * distance):
            for j in range(distance):
                idx1 = k + j
                idx2 = k + j + distance
                if idx1 < signal_size and idx2 < signal_size:
                    t = x[idx2] * tl.exp(-2j * math.pi * j / (2 * distance))
                    x[idx2] = x[idx1] - t
                    x[idx1] = x[idx1] + t
    
    # Apply normalization
    if norm_factor != 1.0:
        x = x * norm_factor
    
    # Store result
    tl.store(output_ptr + offsets * stride, x, mask=mask)

def fftn(input, s=None, dim=None, norm=None, *, out=None):
    """
    Compute N-dimensional FFT using Triton.
    
    Args:
        input: Input tensor
        s: Signal size in transformed dimensions
        dim: Dimensions to transform
        norm: Normalization mode ('forward', 'backward', 'ortho')
        out: Output tensor
        
    Returns:
        Tensor: Transformed tensor
    """
    # Input validation
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    if not input.dtype in [torch.float16, torch.complex32]:
        raise ValueError("Only torch.half and torch.chalf are supported")
    
    # Set default dimensions if not specified
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Set default sizes if not specified
    if s is None:
        s = [input.size(d) for d in dim]
    
    # Verify power of 2 sizes
    for size in s:
        if size & (size - 1) != 0:
            raise ValueError("All transformed dimensions must be powers of 2")
    
    # Calculate normalization factor
    n = math.prod(s)
    if norm == 'forward':
        norm_factor = 1.0 / n
    elif norm == 'ortho':
        norm_factor = 1.0 / math.sqrt(n)
    else:  # 'backward' or None
        norm_factor = 1.0
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Launch kernel for each dimension
    for i, d in enumerate(dim):
        size = s[i]
        BLOCK_SIZE = min(size, 1024)  # Choose appropriate block size
        
        grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
        
        fft_kernel[grid](
            input.data_ptr(),
            out.data_ptr(),
            input.stride(d),
            input.numel(),
            size,
            norm_factor,
            BLOCK_SIZE,
        )
        
        # Update input for next dimension
        if i < len(dim) - 1:
            input = out.clone()
    
    return out
