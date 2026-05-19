import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rand_triton_kernel(
    out_ptr, seed_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load a single seed
    seed = tl.load(seed_ptr)
    # Generate random numbers
    out1, out2, out3, out4 = tl.rand4x(seed, offsets)
    # For simplicity, just use out1
    tl.store(out_ptr + offsets, out1, mask=mask)


def rand(*size,
         generator=None,
         out=None,
         dtype=None,
         layout=torch.strided,
         device=None,
         requires_grad=False,
         pin_memory=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Create output tensor if none is provided
    if out is None:
        out = torch.empty(*size,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          requires_grad=requires_grad,
                          pin_memory=pin_memory)
    else:
        if out.shape != tuple(size):
            raise ValueError("Shape of 'out' does not match the requested size.")

    # Flatten for kernel launch
    total_elems = out.numel()

    # Get seed from generator or create a new one
    if generator is not None:
        # Torch Generator returns 64-bit seed, ensure we handle properly
        seed_val = generator.initial_seed() & 0xffffffff
    else:
        seed_val = torch.randint(0, 2**31, (1,), device=device).item()

    # Create a GPU tensor for seed (single value)
    seed_tensor = torch.tensor([seed_val], dtype=torch.int32, device=device)

    # Launch triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (math.ceil(total_elems / meta['BLOCK_SIZE']),)
    _rand_triton_kernel[grid](
        out,
        seed_tensor,
        total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
