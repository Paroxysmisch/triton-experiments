import torch
import triton
import triton.language as tl
import math

@triton.jit
def fifth_order_fwd_kernel(
    coords_ptr, # pointer to input coordinates (x,y,z)
    out_ptr,    # pointer to output harmonics
    block_size, # number of elements to process per block
    n_elements, # total number of elements
    stride,     # stride for the input coordinates
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and compute the start index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Initialize offsets for x, y, z coordinates
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load coordinates
    x = tl.load(coords_ptr + offsets * stride + 0, mask=mask)
    y = tl.load(coords_ptr + offsets * stride + 1, mask=mask)
    z = tl.load(coords_ptr + offsets * stride + 2, mask=mask)
    
    # Constants for 5th order harmonics
    c1 = 0.31539156525252005  # sqrt(1/(4π))
    c3 = 1.0925484305920792   # sqrt(3/(4π))
    c5 = 1.7701307697799304   # sqrt(5/(4π))
    
    # Compute common terms
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Compute harmonics (l=0)
    h0 = c1 * tl.ones_like(x)
    
    # Compute harmonics (l=1)
    h1_1 = c3 * x
    h1_0 = c3 * z
    h1_1_neg = c3 * y
    
    # Compute harmonics (l=2)
    h2_2 = c5 * xy
    h2_1 = c5 * xz
    h2_0 = c5 * (3.0 * zz - 1.0) / 2.0
    h2_1_neg = c5 * yz
    h2_2_neg = c5 * (xx - yy) / 2.0
    
    # Store results
    base_out_ptr = out_ptr + block_start * 11  # 11 components total
    
    tl.store(base_out_ptr + 0 * n_elements, h0, mask=mask)
    tl.store(base_out_ptr + 1 * n_elements, h1_1, mask=mask)
    tl.store(base_out_ptr + 2 * n_elements, h1_0, mask=mask)
    tl.store(base_out_ptr + 3 * n_elements, h1_1_neg, mask=mask)
    tl.store(base_out_ptr + 4 * n_elements, h2_2, mask=mask)
    tl.store(base_out_ptr + 5 * n_elements, h2_1, mask=mask)
    tl.store(base_out_ptr + 6 * n_elements, h2_0, mask=mask)
    tl.store(base_out_ptr + 7 * n_elements, h2_1_neg, mask=mask)
    tl.store(base_out_ptr + 8 * n_elements, h2_2_neg, mask=mask)

@triton.jit
def fifth_order_bwd_kernel(
    grad_out_ptr,  # pointer to output gradients
    coords_ptr,    # pointer to input coordinates
    grad_in_ptr,   # pointer to input gradients
    block_size,    # number of elements per block
    n_elements,    # total number of elements
    stride,        # stride for coordinates
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load coordinates
    x = tl.load(coords_ptr + offsets * stride + 0, mask=mask)
    y = tl.load(coords_ptr + offsets * stride + 1, mask=mask)
    z = tl.load(coords_ptr + offsets * stride + 2, mask=mask)
    
    # Load gradients
    base_grad_ptr = grad_out_ptr + block_start * 11
    grad_h = [tl.load(base_grad_ptr + i * n_elements, mask=mask) for i in range(9)]
    
    # Constants
    c1 = 0.31539156525252005
    c3 = 1.0925484305920792
    c5 = 1.7701307697799304
    
    # Compute gradients for x, y, z
    grad_x = tl.zeros_like(x)
    grad_y = tl.zeros_like(y)
    grad_z = tl.zeros_like(z)
    
    # Add contributions from each harmonic order
    # l=1 contributions
    grad_x += c3 * grad_h[1]
    grad_y += c3 * grad_h[3]
    grad_z += c3 * grad_h[2]
    
    # l=2 contributions
    grad_x += c5 * (y * grad_h[4] + z * grad_h[5] + x * grad_h[8])
    grad_y += c5 * (x * grad_h[4] + z * grad_h[7] - y * grad_h[8])
    grad_z += c5 * (x * grad_h[5] + y * grad_h[7] + 3.0 * z * grad_h[6])
    
    # Store gradients
    tl.store(grad_in_ptr + offsets * stride + 0, grad_x, mask=mask)
    tl.store(grad_in_ptr + offsets * stride + 1, grad_y, mask=mask)
    tl.store(grad_in_ptr + offsets * stride + 2, grad_z, mask=mask)

class FifthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords):
        # Save input for backward pass
        ctx.save_for_backward(coords)
        
        # Initialize output tensor
        batch_size = coords.shape[0]
        device = coords.device
        output = torch.empty((batch_size, 11), device=device, dtype=coords.dtype)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        fifth_order_fwd_kernel[grid](
            coords.contiguous().data_ptr(),
            output.data_ptr(),
            batch_size,
            batch_size,
            coords.stride(0),
            BLOCK_SIZE=128
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        coords, = ctx.saved_tensors
        batch_size = coords.shape[0]
        grad_input = torch.empty_like(coords)
        
        # Launch backward kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        fifth_order_bwd_kernel[grid](
            grad_output.contiguous().data_ptr(),
            coords.data_ptr(),
            grad_input.data_ptr(),
            batch_size,
            batch_size,
            coords.stride(0),
            BLOCK_SIZE=128
        )
        
        return grad_input

# Wrapper function
def compute_fifth_order_harmonics(coords):
    """
    Compute fifth-order spherical harmonics for given coordinates.
    
    Args:
        coords: torch.Tensor of shape (N, 3) containing x,y,z coordinates
        
    Returns:
        torch.Tensor of shape (N, 11) containing spherical harmonic values
    """
    return FifthOrderSphericalHarmonic.apply(coords)
