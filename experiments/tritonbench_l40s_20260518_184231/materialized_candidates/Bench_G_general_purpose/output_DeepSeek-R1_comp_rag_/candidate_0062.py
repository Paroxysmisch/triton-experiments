import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_warps=8),
    ]
)
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
    thread_idx = tl.arange(0, BLOCK)
    base = pid * BLOCK * 4 + thread_idx * 4
    
    # Generate four random uint32 values using Philox
    r0, r1, r2, r3 = tl.randint4x(philox_seed, philox_offset + (base // 4))
    
    # Convert each to float in [0, 1)
    f0 = tl.libdevice.uint_to_uniform_float(r0)
    f1 = tl.libdevice.uint_to_uniform_float(r1)
    f2 = tl.libdevice.uint_to_uniform_float(r2)
    f3 = tl.libdevice.uint_to_uniform_float(r3)
    
    # Scale to [from_, to)
    scale = to - from_
    f0 = f0 * scale + from_
    f1 = f1 * scale + from_
    f2 = f2 * scale + from_
    f3 = f3 * scale + from_
    
    # Calculate offsets for each of the four elements
    off0 = base + 0
    off1 = base + 1
    off2 = base + 2
    off3 = base + 3
    
    # Store each element with bounds checking
    tl.store(out_ptr + off0, f0, mask=off0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off1, f1, mask=off1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off2, f2, mask=off2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off3, f3, mask=off3 < N, eviction_policy="evict_first")

def philox_cuda_seed_offset() -> (int, int):
    # Helper function to get current Philox seed and offset from PyTorch's CUDA RNG
    state = torch.cuda.get_rng_state()
    seed = state[0:4].view(dtype=torch.int32).item()  # Extract seed
    offset = state[4].item() * 2**64 + state[5].item()  # Philox offset is a 128-bit value, simplified here
    return seed, offset

def uniform_(out: torch.Tensor, from_: float, to: float):
    assert out.is_cuda, "Output tensor must be on CUDA device"
    N = out.numel()
    if N == 0:
        return
    
    seed, offset = philox_cuda_seed_offset()
    
    # Define grid function based on BLOCK size from autotuner
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK'] * 4), )
    
    # Launch kernel with current device context
    with torch.cuda.device(out.device):
        uniform_kernel[grid](
            out.data_ptr(),
            N,
            seed,
            offset,
            from_,
            to,
        )
