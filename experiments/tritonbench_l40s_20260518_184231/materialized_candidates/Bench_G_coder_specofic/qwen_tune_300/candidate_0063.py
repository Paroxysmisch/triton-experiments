import torch
import triton
import triton.language as tl
from flag_gems.utils.random_utils import philox_cuda_seed_offset

@triton.heuristics(
    {
        "num_warps": lambda nargs, **meta: min(
            4, 2 ** (meta["exp2(N/BLOCK):i32"] - 3)
        ),
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
):
    # Generate random numbers using Philox
    r0, r1, r2, r3 = tl.philox(philox_seed, philox_offset)

    # Convert random uints to floats in the range [0, 1)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    # Scale the random numbers to the specified range [from_, to)
    r0 = r0 * (to - from_) + from_
    r1 = r1 * (to - from_) + from_
    r2 = r2 * (to - from_) + from_
    r3 = r3 * (to - from_) + from_

    # Calculate offsets for storing the random numbers
    off_0 = tl.arange(0, BLOCK) + tl.program_id(0) * BLOCK
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    # Store the random numbers, evicting the oldest numbers if necessary
    tl.store(out_ptr + off_0, r0, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, eviction_policy="evict_first")

def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    r"""
    Fills self with uniform random numbers from the range [from_, to).
    """
    N = self.numel()
    exp2_N = max(1, 2 ** (self.numel().bit_length() - 1))
    philox_seed, philox_offset = philox_cuda_seed_offset(N, exp2_N=exp2_N)
    BLOCK = triton.next_power_of_2(N)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"]),)

    with torch.cuda.device(self.device):
        uniform_kernel[grid](self, N, philox_seed, philox_offset, from_, to, BLOCK)
    return self
