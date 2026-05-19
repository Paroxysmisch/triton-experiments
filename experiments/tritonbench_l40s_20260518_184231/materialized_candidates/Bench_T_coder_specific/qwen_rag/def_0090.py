import triton
import triton.language as tl

@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Input with elements randomly zeroed out.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))

@triton.jit
def apply_hard_shrink(x, lambd):
    """
    Applies hard shrinkage to the input.

    Args:
        x: Input tensor.
        lambd: Lambda parameter for hard shrinkage.

    Returns:
        Tensor after applying hard shrinkage.
    """
    return tl.where(tl.abs(x) > lambd, x, 0)

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_ptr, output_ptr, size,
    drop_p, seed, lambd,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses dropout and hard shrinkage operations.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        size: Size of the input tensor.
        drop_p: Dropout probability.
        seed: Seed for dropout mask generation.
        lambd: Lambda parameter for hard shrinkage.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = offsets < size

    input_data = tl.load(input_ptr + offsets, mask=valid_mask)
    dropped_input = apply_dropout(input_data, drop_p, seed, offsets)
    output_data = apply_hard_shrink(dropped_input, lambd)
    tl.store(output_ptr + offsets, output_data, mask=valid_mask)
