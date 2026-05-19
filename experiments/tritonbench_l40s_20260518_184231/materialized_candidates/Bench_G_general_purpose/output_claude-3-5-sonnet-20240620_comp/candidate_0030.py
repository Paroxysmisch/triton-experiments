import triton
import triton.language as tl
import torch

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, in_out_ptr0, in_out_ptr1,
    xnumel, rnumel,
    XBLOCK : tl.constexpr, RBLOCK : tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    rbase = tl.arange(0, RBLOCK)
    rmask = rbase < rnumel
    
    # Load input and compute mean
    mean = tl.zeros([XBLOCK], dtype=tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        r = roffset + rbase
        mask = rmask & (r < rnumel)
        
        # Load input values
        x = tl.load(in_ptr0 + xindex[:, None] * rnumel + r[None, :], 
                   mask=xmask[:, None] & mask[None, :],
                   other=0.0)
        
        # Accumulate sum for mean
        mean += tl.sum(x, axis=1)
    
    mean = mean / rnumel
    
    # Compute variance
    var = tl.zeros([XBLOCK], dtype=tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        r = roffset + rbase
        mask = rmask & (r < rnumel)
        
        x = tl.load(in_ptr0 + xindex[:, None] * rnumel + r[None, :],
                   mask=xmask[:, None] & mask[None, :],
                   other=0.0)
        
        var += tl.sum((x - mean[:, None]) * (x - mean[:, None]), axis=1)
    
    var = var / rnumel
    
    # Compute inverse standard deviation
    inv_std = tl.libdevice.rsqrt(var + 1e-5)
    
    # Store mean and inverse std
    tl.store(in_out_ptr0 + xindex, mean, mask=xmask)
    tl.store(in_out_ptr1 + xindex, inv_std, mask=xmask)
    
    # Normalize input
    for roffset in range(0, rnumel, RBLOCK):
        r = roffset + rbase
        mask = rmask & (r < rnumel)
        
        x = tl.load(in_ptr0 + xindex[:, None] * rnumel + r[None, :],
                   mask=xmask[:, None] & mask[None, :],
                   other=0.0)
        gamma = tl.load(in_ptr1 + r, mask=mask, other=0.0)
        beta = tl.load(in_ptr2 + r, mask=mask, other=0.0)
        
        normalized = (x - mean[:, None]) * inv_std[:, None]
        output = normalized * gamma[None, :] + beta[None, :]
        
        tl.store(out_ptr0 + xindex[:, None] * rnumel + r[None, :],
                output, mask=xmask[:, None] & mask[None, :])

# Wrapper function
def fused_native_layer_norm_no_welford(input_tensor, weight, bias):
    # Get dimensions
    batch_size = input_tensor.size(0)
    hidden_size = input_tensor.size(-1)
    
    # Create output tensors
    output = torch.empty_like(input_tensor)
    mean = torch.empty(batch_size, device=input_tensor.device, dtype=torch.float32)
    inv_std = torch.empty(batch_size, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = lambda meta: (triton.cdiv(batch_size, meta['XBLOCK']),)
    
    # Launch kernel with two different configurations for autotuning
    triton_red_fused_native_layer_norm_no_welford[grid](
        input_tensor, weight, bias, output, mean, inv_std,
        batch_size, hidden_size,
        XBLOCK=128, RBLOCK=32,
    )
    
    return output, mean, inv_std

# Example usage
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'RBLOCK': 32}),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 16}),
    ],
    key=['xnumel', 'rnumel'],
)
def benchmark_layer_norm(batch_size, hidden_size):
    input_tensor = torch.randn(batch_size, hidden_size, device='cuda')
    weight = torch.ones(hidden_size, device='cuda')
    bias = torch.zeros(hidden_size, device='cuda')
    
    output, mean, inv_std = fused_native_layer_norm_no_welford(
        input_tensor, weight, bias
    )
    return output, mean, inv_std
