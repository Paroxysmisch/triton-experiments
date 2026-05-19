import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENT_TYPE: tl.constexpr,
    INT_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    elements = tl.load(input_ptr + offsets, mask=mask, other=0.0, dtype=ELEMENT_TYPE)
    int_elements = elements.view(INT_TYPE)
    sign_bit = (int_elements >> (INT_TYPE.primitive_bitwidth - 1)) & 1
    tl.store(output_ptr + offsets, sign_bit.to(tl.uint8), mask=mask)

def signbit(input, *, out=None):
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    
    device = input.device
    dtype = input.dtype
    
    if device.type != 'cuda':
        input = input.cuda()
    
    # Determine Triton types based on input dtype
    if dtype == torch.float32:
        element_type = tl.float32
        int_type = tl.int32
    elif dtype == torch.float16:
        element_type = tl.float16
        int_type = tl.int16
    elif dtype == torch.float64:
        element_type = tl.float64
        int_type = tl.int64
    else:
        raise ValueError(f"Unsupported dtype {dtype} for signbit")
    
    n_elements = input.numel()
    output = torch.empty(input.size(), dtype=torch.uint8, device=device)
    
    BLOCK_SIZE = 1024  # Adjust based on optimal performance
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    signbit_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMENT_TYPE=element_type,
        INT_TYPE=int_type,
    )
    
    # Convert to boolean tensor
    output = output.to(torch.bool)
    
    # Handle 'out' argument
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a Tensor")
        if out.shape != input.shape:
            raise ValueError("out shape must match input")
        if out.dtype != torch.bool:
            raise ValueError("out must be a bool tensor")
        out.copy_(output)
        return out
    else:
        return output
