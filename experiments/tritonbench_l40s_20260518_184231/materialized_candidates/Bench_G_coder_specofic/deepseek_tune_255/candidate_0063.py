import torch
import triton
import triton.language as tl
from flag_gems.utils.random_utils import philox_cuda_seed_offset

# Triton kernel to generate uniform random numbers
@triton.jit
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
):
    # Constants
    PHILOX_UINT_PER_DIST: tl.constexpr = 4
    eviction_policy = "evict_first"

    # Initialize Philox
    s0, s1, s2, s3 = tl.philox(philox_seed, philox_offset, 0, 0)
    r0, r1, r2, r3 = tl.rand4x(s0, s1, s2, s3)

    # Calculate offsets for storing random numbers
    off_0 = tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    # Generate random floats and scale to [from_, to)
    r0 = r0.to(tl.float32)
    r1 = r1.to(tl.float32)
    r2 = r2.to(tl.float32)
    r3 = r3.to(tl.float32)
    r0 = (r0 / 4294967296.0) * (to - from_) + from_
    r1 = (r1 / 4294967296.0) * (to - from_) + from_
    r2 = (r2 / 4294967296.0) * (to - from_) + from_
    r3 = (r3 / 4294967296.0) * (to - from_) + from_

    # Store random numbers
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy=eviction_policy)
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy=eviction_policy)
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy=eviction_policy)
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy=eviction_policy)

# Function to invoke the Triton kernel
def uniform_(self, from_=0.0, to=1.0):
    # Constants
    PHILOX_UINT_PER_DIST: tl.constexpr = 4
    BLOCK: tl.constexpr = 1024

    # Ensure kernel is executed on the correct CUDA device
    with torch.cuda.device(self.device):
        # Initialize Philox generator state
        philox_seed, philox_offset = philox_cuda_seed_offset(0)
        grid = lambda meta: (triton.cdiv(self.numel(), meta["BLOCK"]),)

        # Invoke the Triton kernel
        uniform_kernel[grid](
            self,
            self.numel(),
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK,
        )

        # Optimize BLOCK and num_warps based on N
        num_warps = 4 if self.numel() < 100000 else 8

        return self
