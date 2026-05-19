import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input, output, p, lambd,
    inplace: tl.constexpr,
    dropout: tl.constexpr,
    hardshrink: tl.constexpr,
):
    # Map the program id to the row of input and output it should compute.
    row = tl.program_id(0)
    input_row_ptr = input + row * input.stride(0)
    output_row_ptr = output + row * output.stride(0)
    row_size = input.stride(0)

    if dropout:
        # Apply dropout
        mask = tl.rand(row_size) > p
        tl.store(output_row_ptr + mask, tl.load(input_row_ptr + mask))
        input_row_ptr = output_row_ptr

    if hardshrink:
        # Apply hard shrinkage
        mask = input_row_ptr >= lambd
        tl.store(output_row_ptr + mask, tl.full(mask.to(tl.int32), 1, mask.dtype))
        mask = input_row_ptr <= -lambd
        tl.store(output_row_ptr + mask, tl.full(mask.to(tl.int32), -1, mask.dtype))

    if inplace:
        # If inplace is True, we need to copy output back to input
        tl.store(input_row_ptr, tl.load(output_row_ptr))

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    output = input if inplace else torch.empty_like(input)

    # grid is defined on the row index of input/output
    def grid(meta): return (triton.cdiv(input.numel(), meta['row_stride']), )

    fused_hardshrink_dropout_kernel[grid](input, output, p, lambd, inplace, training, training)

    if not inplace:
        input = output

    return input
