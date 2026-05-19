import torch
import triton
import triton.language as tl

@triton.jit
def hardshrink_dropout_forward_kernel(
    input_ptr,
    output_ptr,
    size,
    drop_p,
    seed,
    lambd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    # Load input element
    input_val = tl.load(input_ptr + offset, mask=mask)

    # Apply dropout
    random = tl.rand(seed, offset)
    kept = random >= drop_p
    dropout_val = tl.where(kept, input_val / (1 - drop_p), 0.0)

    # Apply hard shrinkage
    hardshrink_mask = (dropout_val < -lambd) | (dropout_val > lambd)
    hardshrink_val = tl.where(hardshrink_mask, dropout_val, 0.0)

    # Store result
    tl.store(output_ptr + offset, hardshrink_val, mask=mask)

def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5
) -> torch.Tensor:
    # Ensure input is contiguous
    input_contig = input.contiguous()
    
    # Check inplace requirements
    if inplace:
        if not input.is_contiguous():
            raise RuntimeError("Input must be contiguous for inplace operation.")
        output = input_contig
    else:
        output = torch.empty_like(input_contig)
    
    # Determine dropout probability based on training mode
    drop_p = p if training else 0.0
    
    # Generate seed for random number generation during training
    if training:
        seed = torch.randint(0, 2**32, (1,), device=input_contig.device).item()
    else:
        seed = 0  # Seed is irrelevant when dropout is disabled
    
    # Launch the Triton kernel
    size = input_contig.numel()
    BLOCK_SIZE = 1024  # Tune this based on hardware for optimal performance
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    hardshrink_dropout_forward_kernel[grid](
        input_contig.data_ptr(),
        output.data_ptr(),
        size,
        drop_p,
        seed,
        lambd,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
