import torch
import triton
import triton.language as tl

def element_wise_kernel_configs():
    return [
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ]

@triton.jit
def apply_dropout(input_val, drop_p, seed, offset):
    random_val = tl.rand(seed, offset)
    return tl.where(random_val < drop_p, 0, input_val / (1 - drop_p))

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def dropout_forward_kernel(
    input_ptr, output_ptr, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(input_ptr + offsets, mask=mask)
    x_drop = apply_dropout(x, drop_p, seed, offsets)
    tl.store(output_ptr + offsets, x_drop, mask=mask)

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def hard_shrink_kernel(
    input_ptr, output_ptr, size,
    lambd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(input_ptr + offsets, mask=mask)
    out = tl.where((x > -lambd) & (x < lambd), 0, x)
    tl.store(output_ptr + offsets, out, mask=mask)

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    x = input
    size = x.numel()
    seed = 42  # Example seed; in practice, can be random or user-defined.

    # Flatten input for Triton
    x_view = x.view(-1)

    if training:
        if inplace:
            # In-place dropout
            dropout_forward_kernel[( (size + 1023) // 1024, )](
                x_view, x_view, size,
                p, seed,
                BLOCK_SIZE=1024,
            )
            drop_output = x_view
        else:
            drop_output = torch.empty_like(x_view)
            dropout_forward_kernel[( (size + 1023) // 1024, )](
                x_view, drop_output, size,
                p, seed,
                BLOCK_SIZE=1024,
            )
    else:
        # No dropout when not training
        drop_output = x_view if inplace else x_view.clone()

    # Apply hard shrink
    hard_shrink_output = torch.empty_like(drop_output)
    hard_shrink_kernel[( (size + 1023) // 1024, )](
        drop_output, hard_shrink_output, size,
        lambd,
        BLOCK_SIZE=1024,
    )

    return hard_shrink_output.view_as(input)
