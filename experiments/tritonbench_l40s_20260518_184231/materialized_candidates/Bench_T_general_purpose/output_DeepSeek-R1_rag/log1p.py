import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(output_ptr, input_ptr, input_numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < input_numel
    x = tl.load(input_ptr + idx, mask=mask)
    y = tl.log1p(x)
    tl.store(output_ptr + idx, y, mask=mask)

def log1p(input, *, out=None):
    # Ensure input is contiguous; if not, create a contiguous copy
    if not input.is_contiguous():
        input = input.contiguous()
    
    # If out is provided, check its properties and handle non-contiguous case
    if out is not None:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
        
        # If out is non-contiguous, compute into a temporary tensor and copy
        if not out.is_contiguous():
            temp_out = torch.empty_like(input)
            log1p_kernel_launch(temp_out, input)
            out.copy_(temp_out)
            return out
    else:
        out = torch.empty_like(input)
    
    # Launch kernel with the contiguous tensors
    log1p_kernel_launch(out, input)
    return out

def log1p_kernel_launch(output, input):
    input_numel = input.numel()
    if input_numel == 0:
        return  # Handle empty tensor case
    BLOCK_SIZE = 1024  # Tune this based on optimal performance
    grid = (triton.cdiv(input_numel, BLOCK_SIZE),)
    log1p_kernel[grid](output, input, input_numel, BLOCK_SIZE=BLOCK_SIZE)
