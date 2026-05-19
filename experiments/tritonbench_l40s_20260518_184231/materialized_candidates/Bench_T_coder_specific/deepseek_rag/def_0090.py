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
def apply_hardshrink(input, lambd):
    """
    Applies hard shrinkage operation to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: The lambda parameter for the hard shrinkage function.

    Returns:
        Input after applying hard shrinkage.
    """
    return tl.where(input < -lambd, input + lambd, tl.where(input > lambd, input - lambd, 0))

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['size'],
)
@triton.jit
def fused_hardshrink_dropout_kernel(
    input_pointer, output_pointer, size,
    drop_p, seed, lambd,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Applies a fused operation consisting of dropout followed by hard shrinkage on the input.

    Args:
        input_pointer: Pointer to the input to perform dropout on.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        lambd: The lambda parameter for the hard shrinkage function.
        BLOCK_SIZE: Block size.
    """
    # This program processes BLOCK_SIZE rows.
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask)
    input = apply_dropout(input, drop_p, seed, offset)
    input = apply_hardshrink(input, lambd)
    tl.store(output_pointer + offset, input, mask=mask)

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5):
    if not inplace:
        input = input.clone()
    if training:
        fused_hardshrink_dropout_kernel[input.numel(),](
            input.data_ptr(), input.data_ptr(),
            input.numel(), p, torch.randint(0, 10000, (1,)).item(), lambd,
            BLOCK_SIZE=64
        )
    else:
        input = apply_hardshrink(input, lambd)
    return input
