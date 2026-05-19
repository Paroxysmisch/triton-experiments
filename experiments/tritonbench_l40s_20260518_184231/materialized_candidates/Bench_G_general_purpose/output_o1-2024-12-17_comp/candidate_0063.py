import triton
import triton.language as tl
import torch


@triton.jit
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK * 4

    # Loop to generate numbers in groups of 4 per thread
    offsets = tl.arange(0, 4)
    offsets += base_offset

    # Each thread handles multiple outputs in a strided manner
    # We compute as many steps as needed so we do not exceed N
    # by adjusting the philox offset.
    # Impl: we'll generate once per block of 4
    r0, r1, r2, r3 = tl.philox(philox_seed, philox_offset + pid)
    random_floats = [
        tl.uint_to_uniform_float(r0),
        tl.uint_to_uniform_float(r1),
        tl.uint_to_uniform_float(r2),
        tl.uint_to_uniform_float(r3),
    ]

    range_size = to - from_
    scaled_values = [from_ + r * range_size for r in random_floats]

    # Conditionally store results
    cond0 = offsets < N
    tl.store(
        out_ptr + offsets[0],
        scaled_values[0],
        mask=cond0[0],
        eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + offsets[1],
        scaled_values[1],
        mask=cond0[1],
        eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + offsets[2],
        scaled_values[2],
        mask=cond0[2],
        eviction_policy="evict_first"
    )
    tl.store(
        out_ptr + offsets[3],
        scaled_values[3],
        mask=cond0[3],
        eviction_policy="evict_first"
    )


def philox_cuda_seed_offset(seed, offset):
    # Placeholder for a more complex method of deriving philox seeds.
    # In practice, you'd combine seed and offset with device-specific ops.
    return seed, offset


def uniform_(out, from_, to, seed, offset):
    # Prepare Philox seed, offset
    philox_seed, philox_offset = philox_cuda_seed_offset(seed, offset)
    N = out.numel()

    # Heuristics for block size and warps
    if N < 65536:
        BLOCK = 128
        num_warps = 4
    else:
        BLOCK = 256
        num_warps = 8

    grid = lambda meta: ((N + 4 * meta["BLOCK"] - 1) // (4 * meta["BLOCK"]),)

    with torch.cuda.device(out.device):
        uniform_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK,
            num_warps=num_warps
        )
