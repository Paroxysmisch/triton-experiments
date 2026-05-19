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
    to,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK * 4 + tl.arange(0, BLOCK) * 4
    off_0 = off
    off_1 = off + 1
    off_2 = off + 2
    off_3 = off + 3
    
    mask_0 = off_0 < N
    mask_1 = off_1 < N
    mask_2 = off_2 < N
    mask_3 = off_3 < N
    
    r0, r1, r2, r3 = tl.rand4x(philox_seed, philox_offset + pid)
    
    f0 = tl.libdevice.uint_to_uniform_float(r0)
    f1 = tl.libdevice.uint_to_uniform_float(r1)
    f2 = tl.libdevice.uint_to_uniform_float(r2)
    f3 = tl.libdevice.uint_to_uniform_float(r3)
    
    scale = to - from_
    scaled0 = from_ + f0 * scale
    scaled1 = from_ + f1 * scale
    scaled2 = from_ + f2 * scale
    scaled3 = from_ + f3 * scale
    
    tl.store(out_ptr + off_0, scaled0, mask=mask_0, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, scaled1, mask=mask_1, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, scaled2, mask=mask_2, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, scaled3, mask=mask_3, eviction_policy="evict_first")

def uniform_(out: torch.Tensor, from_: float, to: float):
    assert out.is_cuda, "Output tensor must be on CUDA device"
    N = out.numel()
    if N == 0:
        return
    
    # Obtain Philox seed and offset
    seed = 0  # Default seed; can be parameterized if needed
    philox_seed, philox_offset = torch.cuda.default_generators[out.device.index].philox_seed(seed)
    philox_offset = philox_offset.cuda(out.device)
    
    # Heuristic to determine block size and number of warps
    max_block = 1024
    BLOCK = min(max_block, triton.next_power_of_2(N // 4))
    if BLOCK < 128:
        BLOCK = 128
    BLOCK = min(BLOCK, 1024)
    
    num_warps = BLOCK // 32 if BLOCK >= 128 else 4
    grid = (triton.cdiv(N, BLOCK * 4),)
    
    with torch.cuda.device(out.device):
        uniform_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            from_,
            to,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
