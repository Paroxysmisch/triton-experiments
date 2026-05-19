import torch
import triton
import triton.language as tl
from triton.language.libdevice import div_rn

@triton.jit
def fused_hstack_div(tensors, divisor, rounding_mode, out):
    # Horizontal stacking
    X = tl.hstack(tensors)
    
    # Element-wise division
    if rounding_mode == 'trunc':
        Y = tl.math.div_rnd(X, divisor, 'trunc')
    elif rounding_mode == 'floor':
        Y = tl.math.div_rnd(X, divisor, 'floor')
    else:
        Y = X / divisor
    
    # Assign result to output tensor
    tl.store(out, Y)

def call_fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Convert divisor to tensor if it is a number
    if isinstance(divisor, (int, float)):
        divisor = torch.tensor(divisor, dtype=tensors[0].dtype, device=tensors[0].device)
    
    # Ensure divisor is broadcastable to the stacked tensor
    assert all(t.shape == tensors[0].shape for t in tensors), "All tensors must have the same shape"
    assert divisor.ndim <= tensors[0].ndim, "Divisor must be broadcastable to the stacked tensor"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(tensors[0], dtype=tensors[0].dtype, device=tensors[0].device)
    
    # Call the Triton kernel
    fused_hstack_div(tensors, divisor, rounding_mode, out)
    
    return out
