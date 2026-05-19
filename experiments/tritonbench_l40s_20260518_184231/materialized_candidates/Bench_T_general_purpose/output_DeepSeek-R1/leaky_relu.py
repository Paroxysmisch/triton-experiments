import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = tl.where(input_data >= 0, input_data, input_data * negative_slope)
    tl.store(output_ptr + offsets, output_data, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        if not input.is_contiguous():
            raise RuntimeError("input must be contiguous for inplace operation")
        output = input
    else:
        if input.is_contiguous():
            input_contig = input
        else:
            input_contig = input.contiguous()
        output_contig = torch.empty_like(input_contig)
        n_elements = input_contig.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        input_flat = input_contig.view(-1)
        output_flat = output_contig.view(-1)
        leaky_relu_kernel[grid](
            input_flat,
            output_flat,
            negative_slope,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        if not input.is_contiguous():
            output = torch.empty_strided(
                input.size(),
                input.stride(),
                dtype=input.dtype,
                device=input.device,
            )
            output.copy_(output_contig)
        else:
            output = output_contig
        return output

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    input_flat = input.view(-1)
    output_flat = output.view(-1)
    leaky_relu_kernel[grid](
        input_flat,
        output_flat,
        negative_slope,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
