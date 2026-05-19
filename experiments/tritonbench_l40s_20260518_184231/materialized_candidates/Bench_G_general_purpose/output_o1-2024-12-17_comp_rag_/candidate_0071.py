import triton
import triton.language as tl
import torch

# ------------------------------------------------------------------------------------
# Kernel function for dropout using a precomputed mask.
@triton.jit
def _triton_dropout(
    x_ptr,         # pointer to the input tensor
    x_keep_ptr,    # pointer to a precomputed mask of 0s and 1s
    output_ptr,    # pointer to the output tensor
    n_elements,    # total elements in the input tensor
    p,             # dropout probability
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and mask
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # Apply dropout using precomputed mask
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # Write back to output tensor
    tl.store(output_ptr + offsets, output, mask=mask)


# ------------------------------------------------------------------------------------
# Kernel function for seeded dropout.
@triton.jit
def _seeded_triton_dropout(
    x_ptr,         # pointer to the input tensor
    output_ptr,    # pointer to the output tensor
    n_elements,    # total elements in the input tensor
    p,             # dropout probability
    seed,          # seed for random number generation
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Generate random mask and apply dropout
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # Write back to output tensor
    tl.store(output_ptr + offsets, output, mask=mask)


# ------------------------------------------------------------------------------------
# Wrapper function for dropout using a precomputed mask.
def triton_dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024):
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x_keep.is_cuda, "Mask tensor must be on CUDA."
    assert x.is_contiguous(), "Input tensor must be contiguous."
    assert x_keep.is_contiguous(), "Mask tensor must be contiguous."
    assert x.shape == x_keep.shape, "Shapes of input tensor and mask must match."

    n_elements = x.numel()
    output = torch.empty_like(x)

    grid = ( (n_elements + block_size - 1) // block_size, )

    _triton_dropout[grid](
        x_ptr=x, 
        x_keep_ptr=x_keep, 
        output_ptr=output, 
        n_elements=n_elements, 
        p=p, 
        BLOCK_SIZE=block_size
    )

    return output


# ------------------------------------------------------------------------------------
# Wrapper function for seeded dropout.
def seeded_triton_dropout(x: torch.Tensor, p: float, seed: int, block_size: int = 1024):
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.is_contiguous(), "Input tensor must be contiguous."

    n_elements = x.numel()
    output = torch.empty_like(x)

    grid = ( (n_elements + block_size - 1) // block_size, )

    _seeded_triton_dropout[grid](
        x_ptr=x, 
        output_ptr=output, 
        n_elements=n_elements, 
        p=p, 
        seed=seed, 
        BLOCK_SIZE=block_size
    )

    return output
