import torch
import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptr += offsets
    weight_ptr += offsets
    output_ptr += offsets

    input = tl.load(input_ptr, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    output = tl.dot(input, weight, allow_tf32=False) + bias
    output = tl.tanh(output)
    tl.store(output_ptr, output, mask=mask)

def tanh_linear(input, weight, bias=None) -> torch.Tensor:
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert bias is None or bias.is_contiguous()

    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(input.device):
        tanh_linear_kernel[grid](input, weight, bias, output, n_elements, BLOCK_SIZE=1024)
    return output
