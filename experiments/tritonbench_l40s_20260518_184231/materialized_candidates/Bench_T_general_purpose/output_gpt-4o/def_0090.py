import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_ptr, output_ptr, mask_ptr,
    n_elements, p, lambd, training, BLOCK_SIZE: tl.constexpr
):
    # Create a block of indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Check if we are within bounds
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout if training
    if training:
        # Generate random mask for dropout
        rand = tl.rand(offsets)
        dropout_mask = rand > p
        input = tl.where(dropout_mask, input / (1.0 - p), 0.0)
        tl.store(mask_ptr + offsets, dropout_mask, mask=mask)
    
    # Apply hard shrinkage
    output = tl.where(tl.abs(input) > lambd, input, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input

    # Flatten the input tensor
    input_flat = input.view(-1)
    n_elements = input_flat.numel()

    # Prepare the mask tensor if training
    mask = None
    if training:
        mask = torch.empty_like(input_flat, dtype=torch.bool)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_hardshrink_dropout_kernel[grid](
        input_flat, output.view(-1), mask,
        n_elements, p, lambd, training,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view_as(input)
