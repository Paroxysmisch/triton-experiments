import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p,
    training,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    x = 1.0 / (1.0 + tl.exp(-x))

    if training:
        rnd = tl.rand(seed, offsets)
        keep = rnd > p
        x = tl.where(keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, x, mask=mask)

def dropout_sigmoid_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    p=0.5,
    training=True,
    inplace=False,
) -> torch.Tensor:
    output = torch.nn.functional.linear(input, weight, bias)
    if inplace:
        output_sigmoid_dropout = output
    else:
        output_sigmoid_dropout = torch.empty_like(output)
    
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    max_seed = 2147483647
    seed = torch.randint(0, max_seed, (1,), device=output.device).item()
    
    _sigmoid_dropout_kernel[grid](
        output,
        output_sigmoid_dropout,
        n_elements,
        p,
        training,
        seed,
        BLOCK_SIZE=1024,
    )
    return output_sigmoid_dropout
