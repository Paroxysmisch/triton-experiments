import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_ptr,
    output_ptr,
    p,
    lambd,
    seed,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    rnd_val = tl.rand(seed, offsets)
    keep_prob = 1.0 - p
    keep = rnd_val < keep_prob
    scale = tl.where(keep_prob > 0.0, 1.0 / keep_prob, 0.0)
    dropout_mask = tl.where(keep, scale, 0.0)
    dropout_output = x * dropout_mask
    hardshrink_output = tl.where(
        (dropout_output > lambd) | (dropout_output < -lambd), dropout_output, 0.0
    )
    tl.store(output_ptr + offsets, hardshrink_output, mask=mask)

def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> torch.Tensor:
    if p < 0 or p > 1:
        raise ValueError("dropout probability must be between 0 and 1")
    
    if input.is_cuda:
        if training:
            if inplace:
                if not input.is_contiguous():
                    raise RuntimeError("inplace operation requires contiguous input")
                output = input
            else:
                output = torch.empty_like(input)
            
            flat_input = input.view(-1)
            flat_output = output.view(-1)
            n_elements = flat_input.numel()
            if n_elements == 0:
                return output
            
            seed = torch.randint(0, 2**32, (1,), device=input.device).item()
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            
            fused_hardshrink_dropout_kernel[grid](
                flat_input, flat_output, p, lambd, seed, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        else:
            return torch.hardshrink(input, lambd=lambd)
    else:
        if training:
            if inplace:
                mask = torch.bernoulli(torch.empty_like(input), 1 - p)
                input.mul_(mask).div_(1 - p)
            else:
                mask = torch.bernoulli(torch.empty_like(input), 1 - p)
                input = input * mask / (1 - p)
            return torch.hardshrink(input, lambd=lambd)
        else:
            return torch.hardshrink(input, lambd=lambd)
