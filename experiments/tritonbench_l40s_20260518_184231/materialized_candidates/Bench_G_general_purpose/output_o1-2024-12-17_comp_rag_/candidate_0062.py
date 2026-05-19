import torch
import triton
import triton.language as tl

@triton.jit
def uniform_kernel(
    out_ptr, 
    N, 
    philox_seed, 
    philox_offset, 
    from_, 
    to_,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK * 4
    range_ = to_ - from_

    # Each thread generates 4 random numbers
    offs_0 = base_offset + tl.arange(0, BLOCK)
    offs_1 = offs_0 + BLOCK
    offs_2 = offs_1 + BLOCK
    offs_3 = offs_2 + BLOCK

    # Generate random numbers
    r0 = tl.rand(philox_seed + 0, offs_0 + philox_offset)
    r1 = tl.rand(philox_seed + 1, offs_1 + philox_offset)
    r2 = tl.rand(philox_seed + 2, offs_2 + philox_offset)
    r3 = tl.rand(philox_seed + 3, offs_3 + philox_offset)

    # Scale into [from_, to_)
    out_0 = from_ + r0 * range_
    out_1 = from_ + r1 * range_
    out_2 = from_ + r2 * range_
    out_3 = from_ + r3 * range_

    # Store results (masking out-of-bounds)
    mask_0 = offs_0 < N
    mask_1 = offs_1 < N
    mask_2 = offs_2 < N
    mask_3 = offs_3 < N

    tl.store(out_ptr + offs_0, out_0, mask=mask_0, evict=True)
    tl.store(out_ptr + offs_1, out_1, mask=mask_1, evict=True)
    tl.store(out_ptr + offs_2, out_2, mask=mask_2, evict=True)
    tl.store(out_ptr + offs_3, out_3, mask=mask_3, evict=True)


def uniform_(out: torch.Tensor, from_: float, to: float, philox_seed: int, philox_offset: int):
    device = out.device
    N = out.numel()
    # Simple heuristic for block size: 256 threads each generating 4 randoms
    BLOCK = 256
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK * 4),)
    # Launch kernel
    with torch.cuda.device(device):
        uniform_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK
        )
