import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    other_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    if other_is_scalar:
        other_val = tl.load(other_ptr)
        other = other_val
    else:
        other = tl.load(other_ptr + offsets, mask=mask)
    product = input * other
    output = tl.maximum(product, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def mul_relu(input, other, inplace=False, out=None):
    if not torch.is_tensor(other):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    
    use_triton = input.is_cuda and other.is_cuda
    if use_triton:
        other_is_scalar = other.dim() == 0 or other.numel() == 1
        if not other_is_scalar:
            if input.shape != other.shape or not input.is_contiguous() or not other.is_contiguous():
                use_triton = False
        
        if use_triton:
            n_elements = input.numel()
            input_flat = input.view(-1)
            other_flat = other.view(-1) if not other_is_scalar else other.reshape(1)
            if inplace:
                if not input_flat.is_contiguous():
                    use_triton = False
                else:
                    output = input_flat
            else:
                output = torch.empty_like(input_flat)
            
            if use_triton:
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                _mul_relu_kernel[grid](
                    input_flat, other_flat, output,
                    n_elements,
                    other_is_scalar,
                    BLOCK_SIZE=1024,
                )
                result = output.view_as(input)
                if out is not None:
                    out.copy_(result)
                    return out
                return result
    
    product = torch.mul(input, other)
    if inplace:
        output = torch.relu_(product)
        if out is not None:
            return out.copy_(output)
        return output
    else:
        output = torch.relu(product)
        if out is not None:
            return out.copy_(output)
        return output
