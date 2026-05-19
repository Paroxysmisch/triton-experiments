import triton
import triton.language as tl

@triton.jit
def gelu_min_kernel(X, OUT, APPROXIMATE, DIM, KEEPDIM, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    if APPROXIMATE == 0:  # 'none'
        cdf = 0.5 * (1.0 + tl.math.erf(x / tl.sqrt(2.0)))
        x = x * cdf
    else:  # 'tanh'
        x_cubed = x * x * x
        inner = 0.044715 * x_cubed + x
        inner = tl.sqrt(2.0 / tl.pi) * inner
        tanh_inner = tl.tanh(inner)
        x = 0.5 * x * (1.0 + tanh_inner)
    
    if DIM is not None:
        # Compute the minimum along the specified dimension
        min_val = tl.min(x, axis=DIM, mask=mask)
        min_idx = tl.argmin(x, axis=DIM, mask=mask)
        tl.store(OUT + offsets, min_val, mask=mask)
        tl.store(OUT + N + offsets, min_idx, mask=mask)
    else:
        # Compute the minimum over all elements
        min_val = tl.min(x, mask=mask)
        min_idx = tl.argmin(x, mask=mask)
        tl.store(OUT, min_val)
        tl.store(OUT + 1, min_idx)

import torch
import triton
import triton.language as tl

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Convert input to a contiguous tensor
    input = input.contiguous()
    N = input.numel()
    
    # Determine the output shape
    if dim is not None:
        out_shape = list(input.shape)
        if not keepdim:
            out_shape[dim] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = (1,)
    
    # Allocate output tensors
    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        indices = torch.empty(out_shape, dtype=torch.long, device=input.device)
    else:
        out, indices = out
    
    # Determine the approximate method
    approximate_val = 0 if approximate == 'none' else 1
    
    # Determine the dimension
    dim_val = dim if dim is not None else -1
    
    # Launch the Triton kernel
    grid = (N // 1024 + 1,)
    gelu_min_kernel[grid](input, out, approximate_val, dim_val, keepdim, N, BLOCK_SIZE=1024)
    
    if dim is not None:
        return out, indices
    else:
        return out

# Example usage
input_tensor = torch.randn(4, 5, device='cuda')
result = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)
print(result)

   result = gelu_min(input_tensor, approximate='none')
   print(result)
   
   result = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)
   print(result)
   
   result = gelu_min(input_tensor, approximate='none', dim=1, keepdim=False)
   print(result)
   
   out_tensor = torch.empty((4, 1), dtype=input_tensor.dtype, device=input_tensor.device)
   indices_tensor = torch.empty((4, 1), dtype=torch.long, device=input_tensor.device)
   result = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True, out=(out_tensor, indices_tensor))
   print(result)
