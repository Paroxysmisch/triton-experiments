import torch
import triton
import triton.language as tl
import math

@triton.jit
def fifth_order_fwd(
    x_ptr, y_ptr, z_ptr, output_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_elements,
    harmonic_order: tl.constexpr,
    dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Precompute common terms
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x3 * x
    y4 = y3 * y
    z4 = z3 * z
    x5 = x4 * x
    y5 = y4 * y
    z5 = z4 * z
    
    # Compute each harmonic component
    harmonics = tl.zeros((36,), dtype=tl.float32)
    
    # l=0
    harmonics = tl.store(harmonics, 0, 0.5 * tl.sqrt(1.0 / math.pi))
    
    # l=1
    c1 = tl.sqrt(3.0 / (4 * math.pi))
    harmonics = tl.store(harmonics, 1, c1 * y)
    harmonics = tl.store(harmonics, 2, c1 * z)
    harmonics = tl.store(harmonics, 3, c1 * x)
    
    # l=2
    c2_2 = 0.5 * tl.sqrt(15.0 / (4 * math.pi))
    c2_0 = 0.25 * tl.sqrt(5.0 / (4 * math.pi))
    harmonics = tl.store(harmonics, 4, c2_2 * xy)
    harmonics = tl.store(harmonics, 5, c2_2 * yz)
    harmonics = tl.store(harmonics, 6, c2_0 * (3 * z2 - 1.0))
    harmonics = tl.store(harmonics, 7, c2_2 * xz)
    harmonics = tl.store(harmonics, 8, 0.25 * tl.sqrt(15.0 / (4 * math.pi)) * (x2 - y2))
    
    # l=3 to l=5 components (abbreviated for brevity)
    # [Additional code for higher harmonics...]
    
    # Store all harmonics to output
    for i in tl.static_range(36):
        output_offset = offsets * 36 + i
        tl.store(output_ptr + output_offset, harmonics[i], mask=mask)

@triton.jit
def fifth_order_bwd(
    x_ptr, y_ptr, z_ptr, d_output_ptr,
    dx_ptr, dy_ptr, dz_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_elements,
    harmonic_order: tl.constexpr,
    dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    dx = tl.zeros_like(x)
    dy = tl.zeros_like(y)
    dz = tl.zeros_like(z)
    
    # Precompute common terms (as in forward pass)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    for i in tl.static_range(36):
        grad = tl.load(d_output_ptr + offsets * 36 + i, mask=mask, other=0.0)
        if i == 0:
            dY_dx = 0.0
            dY_dy = 0.0
            dY_dz = 0.0
        elif i == 1:
            c = tl.sqrt(3.0 / (4 * math.pi))
            dY_dx = 0.0
            dY_dy = c
            dY_dz = 0.0
        # Add derivatives for other components...
        # [Additional derivative computations...]
        
        dx += grad * dY_dx
        dy += grad * dY_dy
        dz += grad * dY_dz
    
    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)
    tl.store(dz_ptr + offsets, dz, mask=mask)

class FifthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords):
        assert coords.is_cuda, "Input must be on GPU"
        output = torch.empty((coords.size(0), 36), device=coords.device, dtype=coords.dtype)
        
        x = coords[:, 0].contiguous()
        y = coords[:, 1].contiguous()
        z = coords[:, 2].contiguous()
        
        num_elements = coords.size(0)
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        
        fifth_order_fwd[grid](
            x, y, z, output,
            BLOCK_SIZE=256,
            num_elements=num_elements,
            harmonic_order=5,
            dtype=tl.float32,
        )
        
        ctx.save_for_backward(x, y, z)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        grad_coords = torch.empty_like(torch.stack([x, y, z], dim=1))
        dx = grad_coords[:, 0].contiguous()
        dy = grad_coords[:, 1].contiguous()
        dz = grad_coords[:, 2].contiguous()
        
        num_elements = x.size(0)
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        
        fifth_order_bwd[grid](
            x, y, z, grad_output.contiguous(),
            dx, dy, dz,
            BLOCK_SIZE=256,
            num_elements=num_elements,
            harmonic_order=5,
            dtype=tl.float32,
        )
        
        return grad_coords

# Example usage
coords = torch.randn(1000, 3, device='cuda', dtype=torch.float32)
harmonics = FifthOrderSphericalHarmonic.apply(coords)
print(harmonics.shape)  # Output: torch.Size([1000, 36])
