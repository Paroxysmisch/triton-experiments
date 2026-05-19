import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(output_ptr, input_ptr, n_elements: tl.constexpr):
    # Each thread computes log1p for one element
    tid = tl.program_id(0)
    if tid < n_elements:
        x = tl.load(input_ptr + tid)
        # Use the identity log(1 + x) ≈ x for small x to improve accuracy
        y = x if x <= 0.5 else tl.log(tl.f32(1) + x)
        tl.store(output_ptr + tid, y)

def log1p(input, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Create an empty output tensor if none is provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the kernel
    log1p_kernel[(n_elements // 256,)](out, input, n_elements)
    
    return out
